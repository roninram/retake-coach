"""The pages and the API, end to end with the stand-in encoder: privacy between tabs, the note, and the visit report."""
import csv

import app as appmod
from conftest import FakeMuse, everyday_png, fundus_jpeg


def check(client, sid="tab-a", name="photo.jpg", data=None, keep_in="visit"):
    r = client.post("/api/check", files={"photo": (name, data or fundus_jpeg(), "image/jpeg")}, data={"keep_in": keep_in, "sid": sid})
    assert r.status_code == 200, r.text
    return r.json()


def rows(log):
    with open(log.path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_pages_load_and_link_to_the_guide(make_client):
    client, _ = make_client()
    for path in ("/", "/batch", "/guide"):
        r = client.get(path)
        assert r.status_code == 200 and 'href="/guide"' in r.text


def test_a_check_returns_a_verdict_and_a_preview(make_client):
    client, _ = make_client()
    r = check(client)
    assert r["verdict"] in ("keep", "retake", "not_fundus")
    assert r["preview"].startswith("data:image/jpeg;base64,")
    assert set(r["checks"]) == {"focus", "aim", "obstruction"}
    assert r["instruction"]


def test_a_file_that_is_not_a_picture_is_refused(make_client):
    client, _ = make_client()
    r = client.post("/api/check", files={"photo": ("notes.txt", b"not a picture", "text/plain")})
    assert r.status_code == 400


def test_the_guard_turns_away_unfamiliar_pictures(make_client, coach):
    coach.g_thr = -1.0                       # nothing looks familiar
    client, _ = make_client()
    r = check(client, data=everyday_png(), name="cat.png")
    assert r["verdict"] == "not_fundus" and r["problems"] == []


def test_each_browser_tab_sees_only_its_own_photos(make_client):
    client, _ = make_client()
    check(client, sid="tab-a")
    assert len(client.get("/api/state", params={"sid": "tab-a"}).json()["visit"]) == 1
    assert client.get("/api/state", params={"sid": "tab-b"}).json()["visit"] == []


def test_no_muse_call_when_nothing_was_a_fundus_photo(make_client):
    muse = FakeMuse()
    client, _ = make_client(muse=muse)
    b = client.post("/api/note", json=dict(results=[dict(verdict="not_fundus")], sid="tab-a")).json()
    assert muse.prompts == [] and b["source"] == "template"
    assert b["covers"].endswith("(left out of the note).")


def test_the_batch_note_does_not_list_missing_views(make_client):
    client, _ = make_client()
    results = [dict(verdict="keep", problems=[], eye="left", field="macula")]
    assert "Kept photos by view" in client.post("/api/note", json=dict(results=results)).json()["facts"]
    assert "Kept photos by view" not in client.post("/api/note", json=dict(results=results, scope="batch")).json()["facts"]


def test_new_patient_adds_one_row_of_counts_and_nothing_identifying(make_client):
    client, log = make_client()
    a = check(client, name="Jane_Doe_1957_OD.jpg")
    b = check(client, name="Jane_Doe_1957_OS.jpg")
    out = client.post("/api/clear", params={"which": "visit", "sid": "tab-a"}).json()
    assert out == dict(cleared="visit", photos=2, saved=True)
    [row] = rows(log)
    assert list(row) == appmod.VisitLog.COLUMNS
    assert "Jane" not in log.path.read_text() and "1957" not in log.path.read_text()
    assert ":" not in row["date"]                                    # a date, never a clock time
    assert int(row["photos_checked"]) == 2
    assert int(row["kept"]) + int(row["sent_back"]) + int(row["not_fundus"]) == 2
    assert int(row["kept"]) == [a["verdict"], b["verdict"]].count("keep")
    assert client.get("/api/state", params={"sid": "tab-a"}).json()["visit"] == []


def test_the_report_keeps_a_note_only_if_it_covers_the_whole_visit(make_client):
    client, log = make_client(muse=FakeMuse())
    first = check(client)
    client.post("/api/note", json=dict(results=[first], sid="tab-a"))
    client.post("/api/clear", params={"which": "visit", "sid": "tab-a"})
    first = check(client)
    client.post("/api/note", json=dict(results=[first], sid="tab-a"))
    check(client)                                                    # a photo after the note: the note is out of date
    client.post("/api/clear", params={"which": "visit", "sid": "tab-a"})
    a, b = rows(log)
    assert a["coach_note"] == "Aim was the most common problem." and b["coach_note"] == ""


def test_an_empty_visit_and_clear_list_add_nothing(make_client):
    client, log = make_client()
    assert client.post("/api/clear", params={"which": "visit", "sid": "empty"}).json()["saved"] is False
    check(client, keep_in="batch")
    assert client.post("/api/clear", params={"which": "batch", "sid": "tab-a"}).json()["saved"] is False
    assert not log.path.exists()


def test_the_report_can_be_downloaded_and_is_off_when_not_set(make_client):
    client, _ = make_client()
    check(client)
    client.post("/api/clear", params={"which": "visit", "sid": "tab-a"})
    r = client.get("/api/report")
    assert r.status_code == 200 and r.text.startswith("date,photos_checked,")
    off, _ = make_client(report=False)
    check(off)
    assert off.post("/api/clear", params={"which": "visit", "sid": "tab-a"}).json()["saved"] is False
    assert off.get("/api/report").status_code == 404
    assert off.get("/api/status").json()["report"] is False
