"""Surgical OOXML insertion, without an Excel reader's load/save round trip.

Good: shift explicit A1 references and comment anchors in a copied package.
Bad: serialize unrelated XML or guess the meaning of INDIRECT or pivot caches.
"""
from __future__ import annotations

import copy
import io
import math
import posixpath
import re
import struct
import zipfile
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape, unescape

from openpyxl.formula import Tokenizer
from openpyxl.formula.tokenizer import TokenizerError

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
TAG = re.compile(r'''<(?:[^<>"']|"[^"]*"|'[^']*')*>''')
ATTR = re.compile(r'''([\w:.-]+)\s*=\s*(["'])(.*?)\2''', re.S)
CELL = re.compile(r"(\$?)([A-Za-z]{1,3})(\$?)([1-9][0-9]*)\Z")
COL = re.compile(r"(\$?)([A-Za-z]{1,3})\Z")
FORMULA = re.compile(
    r"(<(?P<tag>(?:[\w.-]+:)?(?:f|formula|formula1|formula2|definedName))\b[^>]*>)"
    r"(?P<text>[^<]*)(</(?P=tag)\s*>)"
)


class ExportError(ValueError):
    """Safe, fixed diagnostics. Never include source cells in the message."""


def require(condition, message):
    if not condition:
        raise ExportError(message)


def xml(data):
    require(b"<!DOCTYPE" not in data.upper() and b"<!ENTITY" not in data.upper(),
            "XML entity declarations are unsupported")
    try:
        return ET.fromstring(data)
    except ET.ParseError:
        raise ExportError("Malformed workbook XML") from None


def local(tag):
    return tag.rsplit("}", 1)[-1]


def column_number(letters):
    value = 0
    for char in letters.upper():
        value = value * 26 + ord(char) - 64
    require(1 <= value <= 16384, "Column is outside Excel limits")
    return value


def column_name(value):
    require(1 <= value <= 16384, "Insertion would exceed Excel's column limit")
    letters = ""
    while value:
        value, rem = divmod(value - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def shift_ref(value, insert):
    """Shift an A1 cell/range or whitespace-separated ranges; preserve absolutes."""
    def endpoint(part):
        match = CELL.fullmatch(part)
        if match:
            a, letters, b, row = match.groups()
            require(int(row) <= 1048576, "Row is outside Excel limits")
            col = column_number(letters)
            return a + column_name(col + (col >= insert)) + b + row
        match = COL.fullmatch(part)
        if match:
            a, letters = match.groups()
            col = column_number(letters)
            return a + column_name(col + (col >= insert))
        if re.fullmatch(r"\$?[1-9][0-9]*", part):
            return part
        raise ExportError("Unsupported coordinate reference")

    return " ".join(":".join(endpoint(p) for p in group.split(":"))
                    for group in value.split())


def attributes(tag):
    return {m[1]: unescape(m[3], {"&quot;": '"', "&apos;": "'"})
            for m in ATTR.finditer(tag)}


def patch_attributes(tag, changes):
    def replace(match):
        if match[1] not in changes:
            return match[0]
        value = escape(str(changes[match[1]]), {'"': "&quot;", "'": "&apos;"})
        return match[1] + "=" + match[2] + value + match[2]
    return ATTR.sub(replace, tag)


def patch_tags(data, transform):
    def replace(match):
        tag = match[0]
        name = re.match(r"<([\w:.-]+)\b", tag)
        return transform(name[1].split(":")[-1], tag) if name else tag
    return TAG.sub(replace, data.decode("utf-8")).encode("utf-8")


def formula(value, current, insertions):
    """Good: SUM('Team'!$G$2:H9). Bad: INDIRECT builds addresses from text."""
    if not value:
        return value
    try:
        tokens = Tokenizer("=" + value).items
    except TokenizerError:
        raise ExportError("Unsupported formula syntax") from None
    changed = False
    for token in tokens:
        if token.type == "FUNC" and token.subtype == "OPEN":
            require(token.value.upper().rstrip("(").split(".")[-1] not in
                    {"INDIRECT", "HYPERLINK"}, "Dynamic text references are unsupported")
        if token.type != "OPERAND" or token.subtype != "RANGE":
            continue
        original = token.value
        if "!" in original:
            sheet, ref = original.rsplit("!", 1)
            require(not any(x in sheet for x in "[]:"),
                    "External and 3D formula references are unsupported")
            sheet_name = sheet[1:-1].replace("''", "'") if sheet.startswith("'") else sheet
        else:
            sheet, ref, sheet_name = None, original, current
            require(current is not None or not (CELL.fullmatch(ref) or ":" in ref),
                    "Unscoped relative references in global names are unsupported")
        insert = insertions.get(sheet_name.casefold()) if sheet_name else None
        if insert is None:
            continue
        # Bare names resolve through definedNames, which are rewritten separately.
        if not (CELL.fullmatch(ref) or ":" in ref):
            require(sheet is None and "[" not in ref,
                    "Unsupported formula reference on selected sheet")
            continue
        new = shift_ref(ref, insert)
        token.value = (sheet + "!" if sheet is not None else "") + new
        changed |= token.value != original
    return "".join(t.value for t in tokens) if changed else value


def rewrite_formulas(data, current, insertions):
    def replace(match):
        old = unescape(match["text"], {"&quot;": '"', "&apos;": "'"})
        new = formula(old, current, insertions)
        if old == new:
            return match[0]
        attrs = attributes(match[1])
        require(attrs.get("t", "normal") == "normal",
                "Affected shared, array, or data-table formulas are unsupported")
        return match[1] + escape(new) + match[4]
    return FORMULA.sub(replace, data.decode("utf-8")).encode("utf-8")


def part_target(part, target):
    result = posixpath.normpath(posixpath.join(posixpath.dirname(part), target))
    if target.startswith("/"):
        result = target.lstrip("/")
    require(not result.startswith("../"), "Relationship escapes workbook package")
    return result


def sheet_parts(archive):
    book = xml(archive.read("xl/workbook.xml"))
    require(book.tag == "{" + NS + "}workbook", "Only standard .xlsx workbooks are supported")
    rels = {r.attrib["Id"]: r.attrib["Target"]
            for r in xml(archive.read("xl/_rels/workbook.xml.rels"))}
    return [(s.attrib["name"], part_target("xl/workbook.xml", rels[s.attrib["{" + REL + "}id"]]))
            for s in book.find("{" + NS + "}sheets")]


def cell_text(cell, strings):
    if cell is None:
        return ""
    require(cell.find("{" + NS + "}f") is None, "Formula in mapped transcript column is unsupported")
    kind = cell.attrib.get("t", "n")
    if kind == "inlineStr":
        return "".join(t.text or "" for t in cell.iter("{" + NS + "}t"))
    value = cell.find("{" + NS + "}v")
    text = value.text if value is not None and value.text is not None else ""
    if kind == "s":
        require(text.isdigit() and int(text) < len(strings), "Invalid shared string index")
        return strings[int(text)]
    if not text:
        return ""
    require(kind in {"n", "str"}, "Unsupported mapped transcript cell type")
    if kind == "n":
        # Match the frozen converter's finite int/float-to-string contract.
        try:
            number = float(text)
            require(math.isfinite(number), "Non-finite numeric transcript cell")
            if "." not in text and "e" not in text.lower():
                return str(int(text))
            return str(number)
        except ValueError:
            raise ExportError("Invalid numeric transcript cell") from None
    return text


def read_cells(data):
    root = xml(data)
    cells = {}
    rows = set()
    for row in root.findall("{" + NS + "}sheetData/{" + NS + "}row"):
        require(row.attrib.get("r", "").isdigit(), "Missing physical row coordinate")
        number = int(row.attrib["r"])
        require(number not in rows, "Duplicate physical worksheet row")
        rows.add(number)
        last_col = 0
        for cell in row.findall("{" + NS + "}c"):
            ref = cell.attrib.get("r", "")
            match = CELL.fullmatch(ref)
            require(match is not None and int(match[4]) == number, "Invalid physical cell coordinate")
            col = column_number(match[2])
            require(col > last_col and ref not in cells, "Duplicate or unordered worksheet cell")
            cells[ref] = cell
            last_col = col
    return cells, rows


def _worksheet(data, spec, values, insertions):
    insert = spec["coders"][1] + 1
    root = xml(data)
    unsupported = {"mergeCells", "tableParts", "extLst", "pivotTableParts", "sheetProtection",
                   "protectedRanges", "controls", "oleObjects", "customSheetViews", "scenarios"}
    require(not any(local(e.tag) in unsupported for e in root.iter()),
            "Selected sheet has unsupported merges, tables, extensions, protection, or objects")
    supported_children = {"sheetPr", "dimension", "sheetViews", "sheetFormatPr", "cols", "sheetData",
                          "sheetCalcPr", "autoFilter", "sortState", "conditionalFormatting",
                          "dataValidations", "hyperlinks", "printOptions", "pageMargins", "pageSetup",
                          "headerFooter", "drawing", "legacyDrawing", "ignoredErrors", "phoneticPr"}
    require(all(local(e.tag) in supported_children for e in root),
            "Selected sheet has an unsupported worksheet structure")
    for f in root.iter("{" + NS + "}f"):
        require(f.attrib.get("t", "normal") == "normal",
                "Shared, array, or data-table formulas on selected sheets are unsupported")
    ref_tags = {"dimension": "ref", "autoFilter": "ref", "sortState": "ref",
                "sortCondition": "ref", "dataValidation": "sqref",
                "conditionalFormatting": "sqref", "hyperlink": "ref",
                "selection": "sqref", "ignoredError": "sqref"}

    def transform(name, tag):
        attrs, changes = attributes(tag), {}
        if name == "c":
            changes["r"] = shift_ref(attrs["r"], insert)
        if name == "row" and "spans" in attrs:
            require(re.fullmatch(r"[0-9]+:[0-9]+", attrs["spans"]), "Unsupported row spans")
            changes["spans"] = ":".join(str(int(x) + (int(x) >= insert))
                                         for x in attrs["spans"].split(":"))
        if name == "col":
            low, high = int(attrs["min"]), int(attrs["max"])
            changes.update(min=low + (low >= insert), max=high + (high >= insert))
            require(changes["max"] <= 16384, "Column formatting reaches Excel's column limit")
        key = ref_tags.get(name)
        if key and key in attrs:
            changes[key] = shift_ref(attrs[key], insert)
        for key in ("activeCell", "topLeftCell"):
            if key in attrs:
                changes[key] = shift_ref(attrs[key], insert)
        if name == "pane" and "xSplit" in attrs:
            require(attrs.get("state") in {"frozen", "frozenSplit"},
                    "Non-frozen split panes are unsupported")
            split = int(attrs["xSplit"])
            changes["xSplit"] = split + (split >= insert)
        if name == "hyperlink" and "location" in attrs:
            changes["location"] = formula(attrs["location"], spec["sheet"], insertions)
        if name == "filterColumn":
            # A filter crossing the insertion gets one additional physical column.
            filters = list(root.iter("{" + NS + "}autoFilter"))
            require(len(filters) == 1, "Unsupported filter column ownership")
            start = column_number(CELL.fullmatch(filters[0].attrib["ref"].split(":")[0])[2])
            index = int(attrs["colId"])
            changes["colId"] = index + (start < insert <= start + index)
        return patch_attributes(tag, changes)

    data = rewrite_formulas(patch_tags(data, transform), spec["sheet"], insertions)
    # Give the new column the neighboring human column's width/style when explicit.
    cols = root.find("{" + NS + "}cols")
    if cols is not None:
        covered = any(int(c.attrib["min"]) < insert <= int(c.attrib["max"]) for c in cols)
        neighbor = next((c for c in cols if int(c.attrib["min"]) <= insert - 1 <= int(c.attrib["max"])), None)
        if not covered and neighbor is not None:
            text = data.decode()
            col_tags = list(re.finditer(r"<(?:[\w.-]+:)?col\b[^>]*/>", text))
            prefix = re.match(r"<((?:[\w.-]+:)?)col", col_tags[0][0])[1]
            attrs = {"min": str(insert), "max": str(insert), **{
                k: v for k, v in neighbor.attrib.items() if k in {"width", "customWidth", "style", "bestFit"}}}
            tag = "<" + prefix + "col" + "".join(f' {k}="{escape(v)}"' for k, v in attrs.items()) + "/>"
            offset = next((m.start() for m in col_tags if int(attributes(m[0])["min"]) > insert),
                          re.search(r"</(?:[\w.-]+:)?cols>", text).start())
            data = (text[:offset] + tag + text[offset:]).encode()
    # Add one inline string to existing physical rows, retaining every original cell byte.
    row_pattern = re.compile(r"(<(?P<p>(?:[\w.-]+:)?)row\b[^>]*>)(?P<body>.*?)</(?P=p)row>", re.S)
    cells, _ = read_cells(data)
    header_ref = column_name(spec["coders"][1]) + str(spec["header_row"])
    header_style = cells.get(header_ref)

    def add(match):
        row = int(attributes(match[1])["r"])
        text = "AI" if row == spec["header_row"] else values.get(row)
        if text is None:
            return match[0]
        prefix = match["p"]
        style_cell = header_style if row == spec["header_row"] else cells.get(
            column_name(spec["coders"][1]) + str(row))
        style = (' s="' + style_cell.attrib["s"] + '"') if style_cell is not None and "s" in style_cell.attrib else ""
        new_cell = (f'<{prefix}c r="{column_name(insert)}{row}"{style} t="inlineStr">'
                    f'<{prefix}is><{prefix}t xml:space="preserve">{escape(text)}</{prefix}t>'
                    f'</{prefix}is></{prefix}c>')
        body = match["body"]
        for tag in TAG.finditer(body):
            if re.match(r"<(?:[\w.-]+:)?c\b", tag[0]):
                col = column_number(CELL.fullmatch(attributes(tag[0])["r"])[2])
                if col > insert:
                    body = body[:tag.start()] + new_cell + body[tag.start():]
                    break
        else:
            body += new_cell
        return match[1] + body + f'</{prefix}row>'

    return row_pattern.sub(add, data.decode("utf-8")).encode("utf-8")


def _related(data, kind, insert):
    root = xml(data)
    if kind == "documenttask":
        require(len(root) == 0, "Nonempty document tasks on selected sheets are unsupported")
        return data
    if kind in {"comments", "threadedComment"}:
        return patch_tags(data, lambda name, tag: patch_attributes(tag, {
            "ref": shift_ref(attributes(tag)["ref"], insert)}) if "ref" in attributes(tag) else tag)
    if kind == "drawing":
        require(not any(local(e.tag) in {"chart", "graphicFrame", "f"} for e in root.iter()),
                "Chart and formula-bearing drawings on selected sheets are unsupported")
        # SpreadsheetDrawing columns are zero-based. Only anchor markers use xdr:col.
        pattern = re.compile(r"(<(?:[\w.-]+:)?col>)([0-9]+)(</(?:[\w.-]+:)?col>)")
        return pattern.sub(lambda m: m[1] + str(int(m[2]) + (int(m[2]) >= insert - 1)) + m[3],
                           data.decode()).encode()
    if kind == "vmlDrawing":
        require(all(e.attrib.get("ObjectType") == "Note" for e in root.iter()
                    if local(e.tag) == "ClientData"), "Only comment VML anchors are supported")
        pattern = re.compile(r"(<(?:[\w.-]+:)?(?P<name>Anchor|Column)>)([^<]+)(</(?:[\w.-]+:)?(?P=name)>)")
        def anchor(m):
            values = [int(x.strip()) for x in m[3].split(",")]
            require(len(values) == (8 if m["name"] == "Anchor" else 1), "Unsupported VML anchor")
            for index in ([0, 4] if len(values) == 8 else [0]):
                values[index] += values[index] >= insert - 1
            text = ", ".join(map(str, values))
            return m[0] if text == m[3] else m[1] + text + m[4]
        return pattern.sub(anchor, data.decode()).encode()
    raise ExportError("Unsupported related workbook part")


def transform(archive, specs, values):
    """Return only changed part payloads; unrelated parts remain raw ZIP records."""
    parts = sheet_parts(archive)
    by_sheet = dict(parts)
    insertions = {s["sheet"].casefold(): s["coders"][1] + 1 for s in specs}
    require(len(insertions) == len(specs), "Duplicate selected sheets")
    names = archive.namelist()
    require(not any(x.startswith(("_xmlsignatures/", "xl/externalLinks/", "xl/pivot", "xl/slicer"))
                    or x in {"xl/calcChain.xml", "xl/vbaProject.bin"} for x in names),
            "Signed, macro, external-link, pivot, slicer, and calculation-chain packages are unsupported")
    changed = {}
    related_owners = {}
    for spec in specs:
        require(spec["sheet"] in by_sheet, "Selected sheet is absent from workbook")
        part = by_sheet[spec["sheet"]]
        changed[part] = _worksheet(archive.read(part), spec, values[spec["sheet"]], insertions)
        relpath = posixpath.join(posixpath.dirname(part), "_rels", posixpath.basename(part) + ".rels")
        if relpath not in names:
            continue
        for rel in xml(archive.read(relpath)):
            kind = rel.attrib["Type"].rsplit("/", 1)[-1]
            if kind in {"hyperlink", "printerSettings"}:
                continue
            require(kind in {"comments", "threadedComment", "documenttask", "drawing", "vmlDrawing"}
                    and rel.attrib.get("TargetMode") != "External", "Unsupported selected sheet relationship")
            target = part_target(part, rel.attrib["Target"])
            require(target not in related_owners, "Shared drawing/comment ownership is unsupported")
            related_owners[target] = spec["sheet"]
            changed[target] = _related(archive.read(target), kind, spec["coders"][1] + 1)
    for name, part in parts:
        if part not in changed:
            changed[part] = rewrite_formulas(archive.read(part), name, insertions)
    # Names may be local to a sheet. Patch each independently without reserializing workbook.xml.
    book = archive.read("xl/workbook.xml")
    pattern = re.compile(r"<(?P<t>(?:[\w.-]+:)?definedName)\b[^>]*>[^<]*</(?P=t)>")
    def name_formula(m):
        attrs = attributes(m[0].split(">", 1)[0] + ">")
        index = int(attrs["localSheetId"]) if "localSheetId" in attrs else None
        require(index is None or 0 <= index < len(parts), "Invalid defined-name sheet scope")
        return rewrite_formulas(m[0].encode(), parts[index][0] if index is not None else None,
                                insertions).decode()
    changed["xl/workbook.xml"] = pattern.sub(name_formula, book.decode()).encode()
    for name in names:
        if name.startswith("xl/charts/") and name.endswith(".xml"):
            candidate = rewrite_formulas(archive.read(name), None, insertions)
            require(candidate == archive.read(name), "Charts referring to selected sheets are unsupported")
    return {name: data for name, data in changed.items() if data != archive.read(name)}


def raw_zip(source, changes):
    """Good: copy untouched compressed records. Bad: recompress the entire workbook.

    Supports ordinary single-disk ZIP32 .xlsx packages and fails before publishing
    on ZIP64, encryption, duplicate entries, or prefixed/self-extracting archives.
    """
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        infos = archive.infolist()
        require(len(infos) == len({i.filename for i in infos}), "Duplicate ZIP members")
        require(infos and min(i.header_offset for i in infos) == 0 and len(infos) < 65535,
                "Only ordinary ZIP32 workbook packages are supported")
        ordered = sorted(infos, key=lambda i: i.header_offset)
        ends = {info.filename: (ordered[n + 1].header_offset if n + 1 < len(ordered)
                                else archive.start_dir) for n, info in enumerate(ordered)}
        central, position = {}, archive.start_dir
        for info in infos:
            require(not info.flag_bits & 1 and info.extract_version < 45,
                    "Encrypted and ZIP64 members are unsupported")
            require(source[position:position + 4] == b"PK\x01\x02", "Unsupported ZIP directory")
            sizes = struct.unpack_from("<3H", source, position + 28)
            length = 46 + sum(sizes)
            central[info.filename] = source[position:position + length]
            position += length
        require(source[position:position + 4] == b"PK\x05\x06", "Unsupported ZIP ending")
        disks = struct.unpack_from("<2H", source, position + 4)
        require(disks == (0, 0), "Multi-disk ZIP archives are unsupported")
        output, directory = bytearray(), bytearray()
        for info in infos:
            record = central[info.filename]
            if info.filename in changes:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as single:
                    single.writestr(copy.copy(info), changes[info.filename])
                blob = buffer.getvalue()
                with zipfile.ZipFile(io.BytesIO(blob)) as single:
                    end = single.start_dir
                raw = blob[:end]
                record = blob[end:len(blob) - 22]
            else:
                raw = source[info.header_offset:ends[info.filename]]
            require(len(output) < 2**32, "ZIP32 size limit exceeded")
            directory.extend(record[:42] + struct.pack("<I", len(output)) + record[46:])
            output.extend(raw)
        start = len(output)
        output.extend(directory)
        require(len(output) < 2**32, "ZIP32 size limit exceeded")
        output.extend(struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, len(infos), len(infos),
                                  len(directory), start, len(archive.comment)))
        output.extend(archive.comment)
        return bytes(output)
