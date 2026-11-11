use std::io::{Cursor, Read, Write};

use calamine::{open_workbook_auto_from_rs, Data, Reader};
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::reader::Reader as XmlReader;
use quick_xml::writer::Writer as XmlWriter;
use serde::{Deserialize, Serialize};
use zip::write::SimpleFileOptions;
use zip::{ZipArchive, ZipWriter};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportedDocument {
    pub file_name: String,
    pub format: String, // "docx" | "xlsx" | "pdf" | "odt" | "text"
    pub text: String,
    pub summary: String,
    pub can_export_docx: bool,
    pub can_export_xlsx: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplacementPair {
    pub original: String,
    pub replacement: String,
}

pub fn import_document_bytes(file_name: &str, bytes: &[u8]) -> Result<ImportedDocument, String> {
    let ext = file_name
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "docx" => {
            let (text, paras) = extract_docx_text(bytes)?;
            Ok(ImportedDocument {
                file_name: file_name.to_string(),
                format: "docx".to_string(),
                text,
                summary: format!("Word-dokument (.docx) · {paras} stycken/rader inlästa lokalt"),
                can_export_docx: true,
                can_export_xlsx: false,
            })
        }
        "odt" => {
            let (text, paras) = extract_odt_text(bytes)?;
            Ok(ImportedDocument {
                file_name: file_name.to_string(),
                format: "odt".to_string(),
                text,
                summary: format!("OpenDocument (.odt) · {paras} stycken inlästa lokalt"),
                can_export_docx: false,
                can_export_xlsx: false,
            })
        }
        "xlsx" | "xls" | "xlsm" | "xlsb" | "ods" => {
            let (text, sheets, rows) = extract_spreadsheet_text(bytes)?;
            let can_xlsx = ext == "xlsx" || ext == "xlsm";
            Ok(ImportedDocument {
                file_name: file_name.to_string(),
                format: "xlsx".to_string(),
                text,
                summary: format!(
                    "Kalkylblad (.{ext}) · {sheets} flik(ar), {rows} rader inlästa lokalt"
                ),
                can_export_docx: false,
                can_export_xlsx: can_xlsx,
            })
        }
        "pdf" => {
            let text = extract_pdf_text(bytes)?;
            let char_count = text.chars().count();
            Ok(ImportedDocument {
                file_name: file_name.to_string(),
                format: "pdf".to_string(),
                text,
                summary: format!("PDF-dokument (.pdf) · {char_count} tecken extraherade lokalt"),
                can_export_docx: false,
                can_export_xlsx: false,
            })
        }
        _ => {
            let text = decode_text_with_swedish_fallback(bytes);
            let lines = text.lines().count();
            Ok(ImportedDocument {
                file_name: file_name.to_string(),
                format: "text".to_string(),
                text,
                summary: format!("Textfil · {lines} rader inlästa lokalt"),
                can_export_docx: false,
                can_export_xlsx: false,
            })
        }
    }
}

/// Avkodar UTF-8, med automatisk fallback till Windows-1252 / ISO-8859-1 för svenska
/// CSV/TXT-filer exporterade från äldre Windows-program eller Excel.
fn decode_text_with_swedish_fallback(bytes: &[u8]) -> String {
    // Strip UTF-8 BOM om den finns
    let bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    if let Ok(utf8) = std::str::from_utf8(bytes) {
        return utf8.to_string();
    }
    // Windows-1252 / Latin-1 fallback
    bytes.iter().map(|&b| cp1252_byte_to_char(b)).collect()
}

fn cp1252_byte_to_char(b: u8) -> char {
    match b {
        0x80 => '€',
        0x82 => '‚',
        0x83 => 'ƒ',
        0x84 => '„',
        0x85 => '…',
        0x86 => '†',
        0x87 => '‡',
        0x88 => 'ˆ',
        0x89 => '‰',
        0x8A => 'Š',
        0x8B => '‹',
        0x8C => 'Œ',
        0x8E => 'Ž',
        0x91 => '‘',
        0x92 => '’',
        0x93 => '“',
        0x94 => '”',
        0x95 => '•',
        0x96 => '–',
        0x97 => '—',
        0x98 => '˜',
        0x99 => '™',
        0x9A => 'š',
        0x9B => '›',
        0x9C => 'œ',
        0x9E => 'ž',
        0x9F => 'Ÿ',
        other => other as char,
    }
}

/// Extraherar text ur Word (.docx) inkl. sidhuvud, brödtext, tabeller, fotnoter och sidfot.
fn extract_docx_text(bytes: &[u8]) -> Result<(String, usize), String> {
    let cursor = Cursor::new(bytes);
    let mut archive = ZipArchive::new(cursor)
        .map_err(|e| format!("Kunde inte läsa .docx-arkiv (är filen skadad?): {e}"))?;

    let mut xml_parts: Vec<String> = Vec::new();
    for i in 0..archive.len() {
        if let Ok(f) = archive.by_index(i) {
            let name = f.name().to_string();
            if name == "word/document.xml"
                || name.starts_with("word/header")
                || name.starts_with("word/footer")
                || name == "word/footnotes.xml"
                || name == "word/endnotes.xml"
            {
                xml_parts.push(name);
            }
        }
    }

    // Sortera så att header kommer först, sedan document.xml, sedan footnotes/footer
    xml_parts.sort_by_key(|name| {
        if name.starts_with("word/header") {
            0
        } else if name == "word/document.xml" {
            1
        } else if name == "word/footnotes.xml" || name == "word/endnotes.xml" {
            2
        } else {
            3
        }
    });

    if xml_parts.is_empty() {
        return Err("Hittade ingen word/document.xml i .docx-filen.".to_string());
    }

    let mut all_paragraphs: Vec<String> = Vec::new();

    for part_name in xml_parts {
        let mut file = archive
            .by_name(&part_name)
            .map_err(|e| format!("Kunde inte läsa {part_name}: {e}"))?;
        let mut xml_content = String::new();
        file.read_to_string(&mut xml_content)
            .map_err(|e| format!("Kunde inte avkoda {part_name}: {e}"))?;

        let part_paras = parse_word_xml_paragraphs(&xml_content)?;
        all_paragraphs.extend(part_paras);
    }

    let count = all_paragraphs.len();
    let joined = all_paragraphs.join("\n\n");
    Ok((joined, count))
}

fn parse_word_xml_paragraphs(xml: &str) -> Result<Vec<String>, String> {
    let mut reader = XmlReader::from_str(xml);
    reader.config_mut().trim_text(false);

    let mut paragraphs: Vec<String> = Vec::new();
    let mut current_para = String::new();
    let mut current_row_cells: Vec<String> = Vec::new();
    let mut in_table_cell = false;
    let mut in_text_node = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                let local = e.local_name();
                match local.as_ref() {
                    b"t" | b"delText" => {
                        in_text_node = true;
                    }
                    b"tc" => {
                        in_table_cell = true;
                    }
                    b"tr" => {
                        current_row_cells.clear();
                    }
                    b"tab" => {
                        current_para.push('\t');
                    }
                    b"br" | b"cr" => {
                        current_para.push('\n');
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => {
                let local = e.local_name();
                match local.as_ref() {
                    b"tab" => current_para.push('\t'),
                    b"br" | b"cr" => current_para.push('\n'),
                    _ => {}
                }
            }
            Ok(Event::Text(e)) => {
                if in_text_node {
                    if let Ok(txt) = e.unescape() {
                        current_para.push_str(&txt);
                    }
                }
            }
            Ok(Event::End(e)) => {
                let local = e.local_name();
                match local.as_ref() {
                    b"t" | b"delText" => {
                        in_text_node = false;
                    }
                    b"p" => {
                        let trimmed = current_para.trim();
                        if !trimmed.is_empty() {
                            if in_table_cell {
                                current_row_cells.push(trimmed.replace('\n', " "));
                            } else {
                                paragraphs.push(trimmed.to_string());
                            }
                        }
                        current_para.clear();
                    }
                    b"tc" => {
                        in_table_cell = false;
                    }
                    b"tr" => {
                        if !current_row_cells.is_empty() {
                            paragraphs.push(current_row_cells.join(" | "));
                            current_row_cells.clear();
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML-tolkningsfel i Word-dokument: {e}")),
            _ => {}
        }
    }

    Ok(paragraphs)
}

/// Skapar ett nytt maskerat Word-dokument (.docx) direkt från originalets .docx-bytes
/// genom att ersätta alla angivna strängar i dokumentets XML-textnoder med bibehållen formatering.
pub fn export_masked_docx_bytes(
    original_docx_bytes: &[u8],
    replacements: &[ReplacementPair],
) -> Result<Vec<u8>, String> {
    let reader_cursor = Cursor::new(original_docx_bytes);
    let mut archive = ZipArchive::new(reader_cursor)
        .map_err(|e| format!("Kunde inte öppna käll-docx för export: {e}"))?;

    let mut out_buf: Vec<u8> = Vec::with_capacity(original_docx_bytes.len());
    {
        let out_cursor = Cursor::new(&mut out_buf);
        let mut writer = ZipWriter::new(out_cursor);

        // Sortera ersättningar med längsta originaltexten först så att delsträngar inte krockar
        let mut sorted_reps: Vec<&ReplacementPair> = replacements
            .iter()
            .filter(|r| !r.original.is_empty())
            .collect();
        sorted_reps.sort_by(|a, b| b.original.len().cmp(&a.original.len()));

        for i in 0..archive.len() {
            let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
            let name = file.name().to_string();
            let options = SimpleFileOptions::default()
                .compression_method(file.compression());

            let is_word_xml = name == "word/document.xml"
                || name.starts_with("word/header")
                || name.starts_with("word/footer")
                || name == "word/footnotes.xml"
                || name == "word/endnotes.xml";

            if is_word_xml {
                let mut xml_str = String::new();
                file.read_to_string(&mut xml_str)
                    .map_err(|e| format!("Kunde inte läsa {name}: {e}"))?;

                let masked_xml = replace_text_in_word_xml(&xml_str, &sorted_reps)?;
                writer
                    .start_file(&name, options)
                    .map_err(|e| e.to_string())?;
                writer
                    .write_all(masked_xml.as_bytes())
                    .map_err(|e| e.to_string())?;
            } else {
                let mut raw_bytes = Vec::new();
                file.read_to_end(&mut raw_bytes).map_err(|e| e.to_string())?;
                writer
                    .start_file(&name, options)
                    .map_err(|e| e.to_string())?;
                writer.write_all(&raw_bytes).map_err(|e| e.to_string())?;
            }
        }

        writer.finish().map_err(|e| e.to_string())?;
    }

    Ok(out_buf)
}

fn replace_text_in_word_xml(
    xml: &str,
    replacements: &[&ReplacementPair],
) -> Result<String, String> {
    replace_text_in_container_xml(xml, b"p", b"t", replacements)
}

/// Tvåstegs XML-ersättare som hanterar både enskilda `<t>`/`<w:t>`-noder OCH ord som
/// Word/Excel har delat upp över flera `<w:r><w:t>`-runs inuti samma `<w:p>` eller `<si>`.
fn replace_text_in_container_xml(
    xml: &str,
    container_local_name: &[u8],
    text_local_name: &[u8],
    replacements: &[&ReplacementPair],
) -> Result<String, String> {
    // Steg 1: samla vilka container-index (t.ex. <w:p> eller <si>) som har en matchning
    // uppdelad över flera <w:t>/<t>-noder efter att enkla nod-ersättningar gjorts.
    let mut split_containers: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut consolidated_texts: std::collections::HashMap<usize, String> =
        std::collections::HashMap::new();

    {
        let mut reader = XmlReader::from_str(xml);
        reader.config_mut().trim_text(false);
        let mut container_idx = 0usize;
        let mut in_container = false;
        let mut in_text = false;
        let mut texts_in_container: Vec<String> = Vec::new();

        loop {
            match reader.read_event() {
                Ok(Event::Start(e)) => {
                    let local = e.local_name();
                    if local.as_ref() == container_local_name {
                        in_container = true;
                        texts_in_container.clear();
                    } else if in_container && local.as_ref() == text_local_name {
                        in_text = true;
                    }
                }
                Ok(Event::Text(e)) => {
                    if in_container && in_text {
                        let unescaped = e.unescape().map_err(|e| e.to_string())?;
                        let mut single = unescaped.into_owned();
                        for rep in replacements {
                            if single.contains(&rep.original) {
                                single = single.replace(&rep.original, &rep.replacement);
                            }
                        }
                        texts_in_container.push(single);
                    }
                }
                Ok(Event::End(e)) => {
                    let local = e.local_name();
                    if local.as_ref() == text_local_name {
                        in_text = false;
                    } else if local.as_ref() == container_local_name && in_container {
                        in_container = false;
                        if texts_in_container.len() > 1 {
                            let combined = texts_in_container.join("");
                            let mut replaced = combined.clone();
                            for rep in replacements {
                                if replaced.contains(&rep.original) {
                                    replaced = replaced.replace(&rep.original, &rep.replacement);
                                }
                            }
                            if replaced != combined {
                                split_containers.insert(container_idx);
                                consolidated_texts.insert(container_idx, replaced);
                            }
                        }
                        container_idx += 1;
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(format!("XML-analysfel vid export: {e}")),
                _ => {}
            }
        }
    }

    // Steg 2: skriv ut XML med antingen individuell nod-ersättning eller konsoliderad container-text
    let mut reader = XmlReader::from_str(xml);
    reader.config_mut().trim_text(false);

    let mut writer = XmlWriter::new(Cursor::new(Vec::with_capacity(xml.len())));
    let mut container_idx = 0usize;
    let mut in_container = false;
    let mut in_wt = false;
    let mut wt_index_in_container = 0usize;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                let local = e.local_name();
                if local.as_ref() == container_local_name {
                    in_container = true;
                    wt_index_in_container = 0;
                }
                if local.as_ref() == text_local_name {
                    in_wt = true;
                    let start_name = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                    let mut elem = BytesStart::new(start_name);
                    let mut has_space = false;
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"xml:space" {
                            has_space = true;
                        }
                        elem.push_attribute(attr);
                    }
                    if !has_space {
                        elem.push_attribute(("xml:space", "preserve"));
                    }
                    writer
                        .write_event(Event::Start(elem))
                        .map_err(|e| e.to_string())?;
                } else {
                    writer
                        .write_event(Event::Start(e))
                        .map_err(|e| e.to_string())?;
                }
            }
            Ok(Event::Text(e)) => {
                if in_wt {
                    if in_container && split_containers.contains(&container_idx) {
                        if wt_index_in_container == 0 {
                            if let Some(cons) = consolidated_texts.get(&container_idx) {
                                writer
                                    .write_event(Event::Text(BytesText::new(cons)))
                                    .map_err(|e| e.to_string())?;
                            }
                        }
                    } else {
                        let unescaped = e.unescape().map_err(|e| e.to_string())?;
                        let mut modified = unescaped.into_owned();
                        for rep in replacements {
                            if modified.contains(&rep.original) {
                                modified = modified.replace(&rep.original, &rep.replacement);
                            }
                        }
                        writer
                            .write_event(Event::Text(BytesText::new(&modified)))
                            .map_err(|e| e.to_string())?;
                    }
                } else {
                    writer
                        .write_event(Event::Text(e))
                        .map_err(|e| e.to_string())?;
                }
            }
            Ok(Event::End(e)) => {
                let local = e.local_name();
                if local.as_ref() == text_local_name {
                    in_wt = false;
                    wt_index_in_container += 1;
                } else if local.as_ref() == container_local_name && in_container {
                    in_container = false;
                    container_idx += 1;
                }
                let end_name = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                writer
                    .write_event(Event::End(BytesEnd::new(end_name)))
                    .map_err(|e| e.to_string())?;
            }
            Ok(Event::Eof) => break,
            Ok(other) => {
                writer.write_event(other).map_err(|e| e.to_string())?;
            }
            Err(e) => return Err(format!("XML-fel vid dokumentexport: {e}")),
        }
    }

    let bytes = writer.into_inner().into_inner();
    String::from_utf8(bytes).map_err(|e| e.to_string())
}

/// Skapar ett nytt maskerat Excel-dokument (.xlsx) direkt från originalets .xlsx-bytes
/// genom att ersätta aktiva träffar i `xl/sharedStrings.xml` och `xl/worksheets/sheet*.xml`
/// med bibehållna flikar, formler, kolumnbredder och cellformat.
pub fn export_masked_xlsx_bytes(
    original_xlsx_bytes: &[u8],
    replacements: &[ReplacementPair],
) -> Result<Vec<u8>, String> {
    let reader_cursor = Cursor::new(original_xlsx_bytes);
    let mut archive = ZipArchive::new(reader_cursor)
        .map_err(|e| format!("Kunde inte öppna käll-xlsx för export: {e}"))?;

    let mut out_buf: Vec<u8> = Vec::with_capacity(original_xlsx_bytes.len());
    {
        let out_cursor = Cursor::new(&mut out_buf);
        let mut writer = ZipWriter::new(out_cursor);

        let mut sorted_reps: Vec<&ReplacementPair> = replacements
            .iter()
            .filter(|r| !r.original.is_empty())
            .collect();
        sorted_reps.sort_by(|a, b| b.original.len().cmp(&a.original.len()));

        for i in 0..archive.len() {
            let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
            let name = file.name().to_string();
            let options = SimpleFileOptions::default().compression_method(file.compression());

            if name == "xl/sharedStrings.xml" {
                let mut xml_str = String::new();
                file.read_to_string(&mut xml_str)
                    .map_err(|e| format!("Kunde inte läsa {name}: {e}"))?;
                let masked_xml = replace_text_in_container_xml(&xml_str, b"si", b"t", &sorted_reps)?;
                writer
                    .start_file(&name, options)
                    .map_err(|e| e.to_string())?;
                writer
                    .write_all(masked_xml.as_bytes())
                    .map_err(|e| e.to_string())?;
            } else if name.starts_with("xl/worksheets/sheet") && name.ends_with(".xml") {
                let mut xml_str = String::new();
                file.read_to_string(&mut xml_str)
                    .map_err(|e| format!("Kunde inte läsa {name}: {e}"))?;
                let masked_xml = replace_text_in_container_xml(&xml_str, b"is", b"t", &sorted_reps)?;
                writer
                    .start_file(&name, options)
                    .map_err(|e| e.to_string())?;
                writer
                    .write_all(masked_xml.as_bytes())
                    .map_err(|e| e.to_string())?;
            } else {
                let mut raw_bytes = Vec::new();
                file.read_to_end(&mut raw_bytes).map_err(|e| e.to_string())?;
                writer
                    .start_file(&name, options)
                    .map_err(|e| e.to_string())?;
                writer.write_all(&raw_bytes).map_err(|e| e.to_string())?;
            }
        }

        writer.finish().map_err(|e| e.to_string())?;
    }

    Ok(out_buf)
}

/// Extraherar text ur OpenDocument Text (.odt) via content.xml.
fn extract_odt_text(bytes: &[u8]) -> Result<(String, usize), String> {
    let cursor = Cursor::new(bytes);
    let mut archive =
        ZipArchive::new(cursor).map_err(|e| format!("Kunde inte läsa .odt-arkiv: {e}"))?;
    let mut file = archive
        .by_name("content.xml")
        .map_err(|e| format!("Saknar content.xml i .odt-filen: {e}"))?;
    let mut xml = String::new();
    file.read_to_string(&mut xml)
        .map_err(|e| format!("Kunde inte läsa content.xml: {e}"))?;

    let mut reader = XmlReader::from_str(&xml);
    reader.config_mut().trim_text(false);

    let mut paragraphs: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut depth = 0usize;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                let local = e.local_name();
                if local.as_ref() == b"p" || local.as_ref() == b"h" {
                    depth += 1;
                } else if depth > 0 && local.as_ref() == b"tab" {
                    current.push('\t');
                } else if depth > 0 && local.as_ref() == b"line-break" {
                    current.push('\n');
                }
            }
            Ok(Event::Empty(e)) => {
                if depth > 0 {
                    let local = e.local_name();
                    if local.as_ref() == b"tab" {
                        current.push('\t');
                    } else if local.as_ref() == b"line-break" {
                        current.push('\n');
                    }
                }
            }
            Ok(Event::Text(e)) => {
                if depth > 0 {
                    if let Ok(t) = e.unescape() {
                        current.push_str(&t);
                    }
                }
            }
            Ok(Event::End(e)) => {
                let local = e.local_name();
                if (local.as_ref() == b"p" || local.as_ref() == b"h") && depth > 0 {
                    depth -= 1;
                    if depth == 0 {
                        let trimmed = current.trim();
                        if !trimmed.is_empty() {
                            paragraphs.push(trimmed.to_string());
                        }
                        current.clear();
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML-fel i .odt: {e}")),
            _ => {}
        }
    }

    let count = paragraphs.len();
    Ok((paragraphs.join("\n\n"), count))
}

/// Extraherar text ur Excel- och OpenDocument-kalkylblad (.xlsx, .xls, .xlsm, .xlsb, .ods).
fn extract_spreadsheet_text(bytes: &[u8]) -> Result<(String, usize, usize), String> {
    let cursor = Cursor::new(bytes);
    let mut workbook = open_workbook_auto_from_rs(cursor)
        .map_err(|e| format!("Kunde inte läsa kalkylbladet: {e}"))?;

    let sheet_names = workbook.sheet_names().to_vec();
    let mut sections: Vec<String> = Vec::new();
    let mut non_empty_sheets = 0usize;
    let mut total_rows = 0usize;

    for sheet_name in &sheet_names {
        let Ok(range) = workbook.worksheet_range(sheet_name) else {
            continue;
        };

        let mut sheet_rows: Vec<String> = Vec::new();
        for row in range.rows() {
            let cells: Vec<String> = row
                .iter()
                .map(format_cell)
                .filter(|s| !s.is_empty())
                .collect();
            if !cells.is_empty() {
                total_rows += 1;
                sheet_rows.push(cells.join(" | "));
            }
        }

        if !sheet_rows.is_empty() {
            non_empty_sheets += 1;
            let mut section = String::new();
            if sheet_names.len() > 1 {
                section.push_str(&format!("=== Flik: {sheet_name} ===\n\n"));
            }
            // Gruppera rader i mindre block (t.ex. 6 rader per stycke) så att EmbeddingGemma 2
            // kan analysera och rama in känsliga tabellavsnitt med hög precision.
            let blocks: Vec<String> = sheet_rows
                .chunks(6)
                .map(|chunk| chunk.join("\n"))
                .collect();
            section.push_str(&blocks.join("\n\n"));
            sections.push(section);
        }
    }

    if sections.is_empty() {
        return Err("Kalkylbladet verkar vara helt tomt.".to_string());
    }

    Ok((sections.join("\n\n"), non_empty_sheets, total_rows))
}

fn format_cell(cell: &Data) -> String {
    match cell {
        Data::Empty => String::new(),
        Data::String(s) => s.trim().replace('\n', " "),
        Data::Float(f) => {
            if f.fract() == 0.0 && f.abs() < 1e15 {
                format!("{:.0}", f)
            } else {
                format!("{f}")
            }
        }
        Data::Int(i) => i.to_string(),
        Data::Bool(b) => if *b { "Ja" } else { "Nej" }.to_string(),
        Data::DateTime(dt) => dt.to_string(),
        Data::DateTimeIso(s) | Data::DurationIso(s) => s.trim().to_string(),
        Data::Error(e) => format!("{e:?}"),
    }
}

/// Extraherar text ur digitala PDF-dokument (.pdf) i minnet.
fn extract_pdf_text(bytes: &[u8]) -> Result<String, String> {
    let raw = pdf_extract::extract_text_from_mem(bytes)
        .map_err(|e| format!("Kunde inte extrahera text ur PDF-filen: {e}"))?;

    // Städa upp sidbrytningar (\x0c) och radbrytningar till läsbara stycken
    let cleaned = raw
        .replace('\u{000c}', "\n\n")
        .replace("\r\n", "\n")
        .replace('\r', "\n");

    let mut paragraphs: Vec<String> = Vec::new();
    for block in cleaned.split("\n\n") {
        let lines: Vec<&str> = block
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();
        if !lines.is_empty() {
            paragraphs.push(lines.join(" "));
        }
    }

    let result = paragraphs.join("\n\n");
    if result.trim().is_empty() {
        return Err(
            "PDF-filen innehåller ingen markerbar text (det kan vara en inskannad bild-PDF)."
                .to_string(),
        );
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_docx(paragraphs: &[&str]) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let cursor = Cursor::new(&mut buf);
            let mut zip = ZipWriter::new(cursor);
            let options = SimpleFileOptions::default();
            zip.start_file("word/document.xml", options).unwrap();

            let mut xml = String::from(
                r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>"#,
            );
            for p in paragraphs {
                xml.push_str(&format!("<w:p><w:r><w:t>{p}</w:t></w:r></w:p>"));
            }
            xml.push_str("</w:body></w:document>");
            zip.write_all(xml.as_bytes()).unwrap();
            zip.finish().unwrap();
        }
        buf
    }

    #[test]
    fn imports_and_exports_masked_docx_in_memory() {
        let docx_bytes = create_sample_docx(&[
            "Patientjournal för Anders Andersson (19850515-1234).",
            "Kontaktas på anders.andersson@region.se vid frågor.",
        ]);

        let imported = import_document_bytes("journal.docx", &docx_bytes).expect("import docx");
        assert_eq!(imported.format, "docx");
        assert!(imported.can_export_docx);
        assert!(imported.text.contains("Anders Andersson"));
        assert!(imported.text.contains("19850515-1234"));

        let masked_bytes = export_masked_docx_bytes(
            &docx_bytes,
            &[
                ReplacementPair {
                    original: "Anders Andersson".into(),
                    replacement: "[PERSON A]".into(),
                },
                ReplacementPair {
                    original: "19850515-1234".into(),
                    replacement: "[PERSONNUMMER 1]".into(),
                },
                ReplacementPair {
                    original: "anders.andersson@region.se".into(),
                    replacement: "[E-POST 1]".into(),
                },
            ],
        )
        .expect("export masked docx");

        let reimported =
            import_document_bytes("journal_maskerad.docx", &masked_bytes).expect("reimport docx");
        assert!(reimported.text.contains("[PERSON A]"));
        assert!(reimported.text.contains("[PERSONNUMMER 1]"));
        assert!(reimported.text.contains("[E-POST 1]"));
        assert!(!reimported.text.contains("Anders Andersson"));
        assert!(!reimported.text.contains("19850515-1234"));
    }

    #[test]
    fn masks_split_runs_across_wt_nodes_in_docx() {
        let mut buf = Vec::new();
        {
            let cursor = Cursor::new(&mut buf);
            let mut zip = ZipWriter::new(cursor);
            let options = SimpleFileOptions::default();
            zip.start_file("word/document.xml", options).unwrap();
            // Word delar ofta upp ett namn över flera <w:r><w:t>-noder p.g.a. stavningskontroll
            let xml = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>Ansvarig läkare: </w:t></w:r><w:r><w:t>Anders </w:t></w:r><w:proofErr w:type="spellStart"/><w:r><w:t>Andersson</w:t></w:r></w:p>
</w:body></w:document>"#;
            zip.write_all(xml.as_bytes()).unwrap();
            zip.finish().unwrap();
        }

        let masked = export_masked_docx_bytes(
            &buf,
            &[ReplacementPair {
                original: "Anders Andersson".into(),
                replacement: "[PERSON A]".into(),
            }],
        )
        .expect("mask split-run docx");

        let reimported = import_document_bytes("split.docx", &masked).expect("reimport split docx");
        assert_eq!(reimported.text, "Ansvarig läkare: [PERSON A]");
    }
}
