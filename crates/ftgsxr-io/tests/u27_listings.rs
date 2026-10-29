// SPDX-License-Identifier: Apache-2.0

//! U-27 (D-31, `cpu` lane): the three committed listing bodies (`tests/fixtures/listing/`:
//! `index.json`, nginx's JSON listing, Apache's HTML listing of `listing/models/`) give their
//! entries with kinds and sizes; dot-names and the parent link are skipped; names are
//! percent-decoded; `classify_name`. The fields' content per flavor is Q-T17.1's, the
//! refusals Q-T17.2's. Expected values are read off the fixture bodies by hand.

use ftgsxr_io::{
    Entry, EntryKind, classify_name, parse_html_listing, parse_index_manifest, parse_nginx_json,
};
use ftgsxr_testkit::fixtures_dir;
use url::Url;

/// The URL the bodies list: Apache's title is `Index of /listing/models`.
fn base() -> Url {
    Url::parse("http://localhost:8000/listing/models/").unwrap()
}

fn body(name: &str) -> String {
    std::fs::read_to_string(fixtures_dir().join("listing").join(name)).unwrap()
}

fn entry(name: &str, location: &str, kind: EntryKind, size: Option<u64>) -> Entry {
    Entry {
        name: name.to_owned(),
        location: location.to_owned(),
        kind,
        size,
        mtime: None,
        sha256: None,
        thumbnail: None,
        audio: None,
        scene: None,
    }
}

const PROBE: &str = "http://localhost:8000/listing/models/probe%20p1.safetensors";
const PLY: &str = "http://localhost:8000/listing/models/snapshot.ply";
const TINY: &str = "http://localhost:8000/listing/models/tiny/";

#[test]
fn u27_index_manifest_entries_kinds_sizes() {
    let entries = parse_index_manifest(&body("index.json"), &base()).unwrap();
    let mut probe = entry("probe p1", PROBE, EntryKind::Exchange, Some(1092));
    probe.sha256 =
        Some("ffa856e1f1647386714f84d06bd7b82aadd3a30de74509d87bd68985388f636c".to_owned());
    let mut ply = entry("snapshot", PLY, EntryKind::Ply, Some(2022));
    ply.sha256 =
        Some("8529f07a3f7ce303e1130a7961d1aef1b9fbb4a32fa2e46aa91abc7c9fed7f37".to_owned());
    let tiny = entry("tiny", TINY, EntryKind::Checkpoint, Some(19677));
    assert_eq!(entries, vec![probe, ply, tiny]);
}

/// An index of one entry with `fields` added to a valid entry's.
fn index_with(top: &str, fields: &str) -> String {
    format!(
        r#"{{"format": "ftgsxr-index", "version": 1{top}, "entries": [{{"name": "m", "path": "m.ply", "kind": "ply", "size": 5{fields}}}]}}"#
    )
}

#[test]
fn u27_index_manifest_optional_fields_resolved_and_versions() {
    let body = index_with(
        r#", "unknown": true"#,
        r#", "sha256": "FFA856E1F1647386714F84D06BD7B82AADD3A30DE74509D87BD68985388F636C", "thumbnail": "thumbs/m.png", "audio": "/audio/m.ogg", "scene": "https://example.org/m.json", "extra": 1"#,
    );
    let entries = parse_index_manifest(&body, &base()).unwrap();
    let mut m = entry(
        "m",
        PLY.replace("snapshot", "m").as_str(),
        EntryKind::Ply,
        Some(5),
    );
    m.sha256 = Some("ffa856e1f1647386714f84d06bd7b82aadd3a30de74509d87bd68985388f636c".to_owned());
    m.thumbnail = Some("http://localhost:8000/listing/models/thumbs/m.png".to_owned());
    m.audio = Some("http://localhost:8000/audio/m.ogg".to_owned());
    m.scene = Some("https://example.org/m.json".to_owned());
    assert_eq!(entries, vec![m]);

    // The index's own URL resolves alike (§1.11: paths relative to the index's URL).
    let at_file = base().join("index.json").unwrap();
    assert_eq!(
        parse_index_manifest(&index_with("", ""), &at_file).unwrap()[0].location,
        "http://localhost:8000/listing/models/m.ply"
    );
    // A minor version is ignored (§1.11's versioning rule).
    let minor = index_with("", "").replace(r#""version": 1"#, r#""version": "1.3""#);
    assert_eq!(parse_index_manifest(&minor, &base()).unwrap().len(), 1);
}

#[test]
fn u27_index_manifest_refusals() {
    let code = |body: &str| parse_index_manifest(body, &base()).unwrap_err().code();
    assert_eq!(code("<html></html>"), "E-JSON");
    let schema_faults = [
        "[]".to_owned(),
        index_with("", "").replace("ftgsxr-index", "ftgsxr-scene"),
        index_with("", "").replace(r#""version": 1"#, r#""version": 2"#),
        index_with("", "").replace(r#""version": 1"#, r#""version": "2.0""#),
        index_with("", "").replace(r#""kind": "ply""#, r#""kind": "zip""#),
        index_with("", "").replace(r#""size": 5"#, r#""size": -5"#),
        index_with("", "").replace(r#", "size": 5"#, ""),
        index_with("", "").replace(r#""name": "m", "#, ""),
        index_with("", r#", "sha256": "abc""#),
        index_with("", r#", "thumbnail": 3"#),
        r#"{"format": "ftgsxr-index", "version": 1, "entries": {}}"#.to_owned(),
    ];
    for body in &schema_faults {
        assert_eq!(code(body), "E-SCENE-FORMAT", "{body}");
    }
}

#[test]
fn u27_nginx_json_entries_kinds_sizes() {
    let entries = parse_nginx_json(&body("nginx.json"), &base()).unwrap();
    let mtime = Some("Thu, 29 Oct 2026 12:00:00 GMT".to_owned());
    let with_mtime = |mut e: Entry| {
        e.mtime = mtime.clone();
        e
    };
    // `.hidden.safetensors` (a dot-name) and `notes.txt` (not a model) are skipped;
    // `probe p1.safetensors` is percent-encoded into its location.
    assert_eq!(
        entries,
        vec![
            with_mtime(entry("tiny", TINY, EntryKind::Dir, None)),
            with_mtime(entry("probe p1", PROBE, EntryKind::Exchange, Some(1092))),
            with_mtime(entry("snapshot", PLY, EntryKind::Ply, Some(2022))),
        ]
    );
}

#[test]
fn u27_nginx_json_refusals_and_other_types() {
    for body in ["<html></html>", "{}", r#"[{"name": "a.ply"}]"#, r#"[1]"#] {
        assert_eq!(
            parse_nginx_json(body, &base()).unwrap_err().code(),
            "E-NET-LISTING",
            "{body}"
        );
    }
    // `other` entries are skipped; a name with reserved characters stays one segment.
    let body = r#"[{"name": "link.ply", "type": "other"}, {"name": "a#b?c:d.PLY", "type": "file", "size": 3}]"#;
    assert_eq!(
        parse_nginx_json(body, &base()).unwrap(),
        vec![entry(
            "a#b?c:d",
            "http://localhost:8000/listing/models/a%23b%3Fc%3Ad.PLY",
            EntryKind::Ply,
            Some(3)
        )]
    );
    assert_eq!(parse_nginx_json("[]", &base()).unwrap(), vec![]);
}

#[test]
fn u27_html_listing_entries_kinds() {
    let entries = parse_html_listing(&body("apache.html"), &base()).unwrap();
    // The parent link `/listing/`, the column-sort links `?C=N;O=D`, `.hidden.safetensors`
    // and `notes.txt` are skipped; `probe%20p1.safetensors` is percent-decoded into its
    // name; the size column (rounded) gives no size.
    assert_eq!(
        entries,
        vec![
            entry("probe p1", PROBE, EntryKind::Exchange, None),
            entry("snapshot", PLY, EntryKind::Ply, None),
            entry("tiny", TINY, EntryKind::Dir, None),
        ]
    );
}

#[test]
fn u27_classify_name() {
    assert_eq!(classify_name("probe p1.safetensors"), EntryKind::Exchange);
    assert_eq!(classify_name("MODEL.SafeTensors"), EntryKind::Exchange);
    assert_eq!(classify_name("snapshot.ply"), EntryKind::Ply);
    assert_eq!(classify_name("snapshot.PLY"), EntryKind::Ply);
    assert_eq!(classify_name("tiny/"), EntryKind::Dir);
    assert_eq!(classify_name("tiny"), EntryKind::Other);
    assert_eq!(classify_name("notes.txt"), EntryKind::Other);
    assert_eq!(classify_name("snapshot.ply.txt"), EntryKind::Other);
    assert_eq!(classify_name("manifest.json"), EntryKind::Other);
}

#[test]
fn u27_html_listing_nginx_style_links_and_refusal() {
    // nginx's HTML autoindex form: a `../` parent link; also a link elsewhere, a deeper
    // link, an icon link repeating a file, a character reference and a fragment.
    let body = r#"<html><head><title>Index of /listing/models/</title></head><body>
<h1>Index of /listing/models/</h1><hr><pre><a href="../">../</a>
<a href="tiny/">tiny/</a>                                              29-Oct-2026 12:00       -
<a href="a&amp;b.ply"><img src="/icons/a.gif"></a> <a href="a&amp;b.ply#top">a&amp;b.ply</a>
<a href="http://elsewhere.example/listing/models/x.ply">x.ply</a>
<a href="tiny/params.safetensors">deep</a> <a href="/listing/models/m%C3%A9.SAFETENSORS">m</a>
</pre><hr></body></html>"#;
    assert_eq!(
        parse_html_listing(body, &base()).unwrap(),
        vec![
            entry("tiny", TINY, EntryKind::Dir, None),
            entry(
                "a&b",
                "http://localhost:8000/listing/models/a&b.ply",
                EntryKind::Ply,
                None
            ),
            entry(
                "mé",
                "http://localhost:8000/listing/models/m%C3%A9.SAFETENSORS",
                EntryKind::Exchange,
                None
            ),
        ]
    );
    for body in [
        r#"[{"name": "a.ply", "type": "file"}]"#,
        "<p>no anchors</p>",
        "",
    ] {
        assert_eq!(
            parse_html_listing(body, &base()).unwrap_err().code(),
            "E-NET-LISTING",
            "{body}"
        );
    }
}
