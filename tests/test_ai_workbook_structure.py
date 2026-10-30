"""Synthetic OOXML fixtures for structures ordinary load/save would discard."""
import io
import json
import zipfile
from xml.etree import ElementTree as ET

import pytest

from ptg.export.ai_workbook import ExportError, export_profile
from ptg.export._ai_workbook_ooxml import NS, raw_zip
from test_ai_workbook import fixture_files, replace_parts, sync_workbook_hash


def test_shared_strings_threaded_comments_drawing_and_vml_anchors(tmp_path):
    profile, workbook, journal, plan = fixture_files(tmp_path)
    with zipfile.ZipFile(workbook) as z:
        rel = z.read("xl/worksheets/_rels/sheet1.xml.rels").decode()
        # Prefixes intentionally differ. Text resembling references must stay literal.
        rel = rel.replace("</Relationships>", '''
<Relationship Type="http://schemas.microsoft.com/office/2017/10/relationships/threadedComment" Target="../threadedComments/test.xml" Id="threaded"/>
<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" Target="../drawings/test.xml" Id="drawing"/>
<Relationship Type="http://schemas.microsoft.com/office/2019/04/relationships/documenttask" Target="../documenttasks/test.xml" Id="task"/>
</Relationships>''')
        sheet = z.read("xl/worksheets/sheet1.xml").decode()
        # Use a real shared string for the transcript while keeping the frozen text identical.
        sheet = sheet.replace('<c r="D2" t="inlineStr"><is><t>', '<c r="D2" t="s"><v>0</v></c><!--')
        text = plan["bundle"]["rows"][0]["text"]
        sheet = sheet.replace(text + '</t></is></c>', text + ' -->', 1)
        old_vml = next(n for n in z.namelist() if n.endswith(".vml"))
        # Standard synthetic sharedStrings relation is unnecessary for the surgical reader;
        # include it in OPC as a genuine workbook relationship and content type as well.
        workbook_rels = z.read("xl/_rels/workbook.xml.rels").decode().replace(
            "</Relationships>", '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml" Id="strings"/></Relationships>')
        content_types = z.read("[Content_Types].xml").decode().replace(
            "</Types>", '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>')
        entries = {n: z.read(n) for n in z.namelist()}
    strings = f'<sst xmlns="{NS}" count="1" uniqueCount="1"><si><t>{text}</t></si></sst>'.encode()
    entries.update({
        "xl/worksheets/_rels/sheet1.xml.rels": rel.encode(),
        "xl/worksheets/sheet1.xml": sheet.encode(),
        "xl/_rels/workbook.xml.rels": workbook_rels.encode(),
        "[Content_Types].xml": content_types.encode(),
        "xl/sharedStrings.xml": strings,
        "xl/threadedComments/test.xml": b'<tc:ThreadedComments xmlns:tc="http://schemas.microsoft.com/office/spreadsheetml/2018/threadedcomments"><tc:threadedComment ref="G2" id="fictional"><tc:text>G2 is literal</tc:text></tc:threadedComment></tc:ThreadedComments>',
        "xl/drawings/test.xml": b'<x:wsDr xmlns:x="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"><x:twoCellAnchor><x:from><x:col>6</x:col><x:colOff>0</x:colOff><x:row>1</x:row><x:rowOff>0</x:rowOff></x:from><x:to><x:col>8</x:col><x:colOff>0</x:colOff><x:row>5</x:row><x:rowOff>0</x:rowOff></x:to><x:clientData/></x:twoCellAnchor></x:wsDr>',
        "xl/documenttasks/test.xml": b'<Tasks xmlns="http://schemas.microsoft.com/office/tasks/2019/documenttasks"/>',
        old_vml: b'<xml xmlns:x="urn:schemas-microsoft-com:office:excel"><x:ClientData ObjectType="Note"><x:Anchor>6, 15, 1, 0, 8, 15, 5, 0</x:Anchor><x:Column>6</x:Column><x:Row>1</x:Row></x:ClientData></xml>',
    })
    with zipfile.ZipFile(workbook, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in entries.items():
            z.writestr(name, data)
    sync_workbook_hash(workbook, journal, plan)
    export_profile(profile)
    with zipfile.ZipFile(tmp_path / "exports/copied.xlsx") as z:
        assert z.read("xl/sharedStrings.xml") == strings
        tc = z.read("xl/threadedComments/test.xml")
        assert b'ref="H2"' in tc and b'G2 is literal' in tc
        dr = ET.fromstring(z.read("xl/drawings/test.xml"))
        assert [e.text for e in dr.iter() if e.tag.endswith("}col")] == ["7", "9"]
        vml = ET.fromstring(z.read(old_vml))
        assert next(e.text for e in vml.iter() if e.tag.endswith("}Column")) == "7"
        assert next(e.text for e in vml.iter() if e.tag.endswith("}Anchor")) == "7, 15, 1, 0, 9, 15, 5, 0"
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        ai = next(e for e in root.iter("{" + NS + "}c") if e.attrib["r"] == "G2")
        assert ai.attrib["t"] == "inlineStr"


@pytest.mark.parametrize("structure", ["merge", "table", "extension", "indirect", "shared_formula", "array_formula", "global_relative_name", "column_limit"])
def test_unsupported_structures_fail_before_publication(tmp_path, structure):
    profile, workbook, journal, plan = fixture_files(tmp_path)
    with zipfile.ZipFile(workbook) as z:
        sheet = z.read("xl/worksheets/sheet1.xml")
        changes = {}
        if structure == "merge":
            sheet = sheet.replace(b"</worksheet>", b'<mergeCells count="1"><mergeCell ref="G2:H2"/></mergeCells></worksheet>')
        elif structure == "table":
            sheet = sheet.replace(b"</worksheet>", b'<tableParts count="0"/></worksheet>')
        elif structure == "extension":
            sheet = sheet.replace(b"</worksheet>", b'<extLst/></worksheet>')
        elif structure == "indirect":
            sheet = sheet.replace(b'<f>SUM(G4:H4)</f>', b'<f>INDIRECT("G2")</f>')
        elif structure in {"shared_formula", "array_formula"}:
            kind = "shared" if structure == "shared_formula" else "array"
            sheet = sheet.replace(b'<f>SUM(G4:H4)</f>', f'<f t="{kind}" ref="J2:J4">SUM(G4:H4)</f>'.encode())
        elif structure == "global_relative_name":
            changes["xl/workbook.xml"] = z.read("xl/workbook.xml").replace(
                b"'Fictional Team (S26)'!$G$2:$G$4", b"G2:G4")
        else:
            sheet = sheet.replace(b"</cols>", b'<col min="16384" max="16384" width="10"/></cols>')
        changes["xl/worksheets/sheet1.xml"] = sheet
    replace_parts(workbook, changes)
    sync_workbook_hash(workbook, journal, plan)
    with pytest.raises(ExportError):
        export_profile(profile)
    assert not (tmp_path / "exports/copied.xlsx").exists()
    assert not (tmp_path / "exports/copied.xlsx.ai.json").exists()


def test_raw_zip_rejects_duplicate_members_and_preserves_archive_comment():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("one", b"unchanged")
        z.writestr("two", b"change")
        z.comment = b"archive comment"
    result = raw_zip(buffer.getvalue(), {"two": b"new"})
    with zipfile.ZipFile(io.BytesIO(result)) as z:
        assert z.comment == b"archive comment"
        assert z.read("one") == b"unchanged" and z.read("two") == b"new"
    with pytest.warns(UserWarning):
        with zipfile.ZipFile(buffer, "a") as z:
            z.writestr("one", b"duplicate")
    with pytest.raises(ExportError, match="Duplicate ZIP"):
        raw_zip(buffer.getvalue(), {})
