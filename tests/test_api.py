import sys
import os
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.app import app


client = TestClient(app)


def test_normal_input_success():
    payload = {
        "title": "Maybank Reports Record Profits in Kuala Lumpur",
        "body": "Malayan Banking Berhad announced strong financial performance on 12 October 2026."
    }
    response = client.post("/extract-entities", json=payload)


    assert response.status_code == 200
    data = response.json()


    assert "entities" in data
    assert "meta" in data
    assert data["meta"]["entity_count"] > 0
    assert "model" in data["meta"]


    first_ent = data["entities"][0]
    assert "text" in first_ent
    assert "label" in first_ent
    assert "start" in first_ent
    assert "end" in first_ent
    assert isinstance(first_ent["start"], int)
    assert isinstance(first_ent["end"], int)
    assert first_ent["start"] < first_ent["end"]


def test_empty_title_and_body_fails():
    res_empty_title = client.post("/extract-entities", json={"title": "", "body": "Some body"})
    assert res_empty_title.status_code == 400


    res_empty_body = client.post("/extract-entities", json={"title": "Some title", "body": "   "})
    assert res_empty_body.status_code == 400


    res_both_empty = client.post("/extract-entities", json={"title": "   ", "body": ""})
    assert res_both_empty.status_code == 400


def test_non_string_types_fail():
    res_numeric = client.post("/extract-entities", json={"title": 12345, "body": "Some body"})
    assert res_numeric.status_code == 400


    res_none = client.post("/extract-entities", json={"title": "Title", "body": None})
    assert res_none.status_code == 400


def test_input_exceeds_max_length_fails():
    too_long_title = "A" * 1005
    res = client.post("/extract-entities", json={"title": too_long_title, "body": "Normal body"})
    assert res.status_code == 400
    assert "exceeds maximum length" in res.json().get("detail", "")


def test_punctuation_and_repeated_entities():
    payload = {
        "title": "BNM & CIMB Group (Malaysia) Review Lending Rates",
        "body": "Bank Negara Malaysia (BNM) and CIMB Group confirmed new rates today."
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 200
    data = response.json()
    texts = [ent["text"] for ent in data["entities"]]


    assert any("CIMB" in t for t in texts)
    assert any("BNM" in t or "Bank Negara" in t for t in texts)


def test_numbers_and_dates_extraction():
    payload = {
        "title": "Quarterly Monetary Meeting Schedule",
        "body": "The committee will convene on 15 November 2026 to discuss third quarter targets."
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 200
    data = response.json()
    labels = [ent["label"] for ent in data["entities"]]

    assert "DATE" in labels


def test_non_ascii_and_multilingual_text():
    payload = {
        "title": "Yayasan Maybank memperkukuh program pendidikan komuniti",
        "body": "Inisiatif kelestarian ini akan memberi manfaat kepada belia di Sabah dan Sarawak."
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 200
    data = response.json()
    texts = [ent["text"] for ent in data["entities"]]

    assert any("Sabah" in t or "Sarawak" in t or "Maybank" in t for t in texts)


def test_zero_named_entities():
    payload = {
        "title": "A simple guide to healthy habits",
        "body": "Drinking clean water and sleeping early helps improve focus and physical energy."
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["entity_count"] == 0
    assert data["entities"] == []


def test_deterministic_output():
    payload = {
        "title": "Petronas and Shell Expand Offshore Operations",
        "body": "Offshore exploration in Sarawak commences next Monday."
    }
    res1 = client.post("/extract-entities", json=payload).json()
    res2 = client.post("/extract-entities", json=payload).json()
    assert res1["entities"] == res2["entities"]
    assert res1["meta"]["entity_count"] == res2["meta"]["entity_count"]


def test_confidence_and_normalized_fields():
    payload = {
        "title": "BNM reviews interest rates with Maybank in KL",
        "body": "Bank Negara Malaysia discussed guidelines today."
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "confidence_explanation" in data["meta"]


    for ent in data["entities"]:
        assert "confidence" in ent
        assert 0.0 <= ent["confidence"] <= 1.0
        assert "normalized" in ent


    norm_map = {ent["text"]: ent["normalized"] for ent in data["entities"]}
    if "BNM" in norm_map:
        assert norm_map["BNM"] == "Bank Negara Malaysia"


def test_batch_endpoint_success():
    batch_payload = {
        "articles": [
            {
                "title": "CIMB expands into Singapore",
                "body": "CIMB Bank confirmed branch expansion."
            },
            {
                "title": "A guide to fitness",
                "body": "Daily exercise and balanced nutrition boost wellbeing."
            }
        ]
    }
    response = client.post("/extract-entities/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_articles"] == 2
    assert len(data["results"]) == 2

    assert data["results"][0]["meta"]["entity_count"] > 0
    assert data["results"][1]["meta"]["entity_count"] == 0


def test_batch_endpoint_validation():
    res_empty = client.post("/extract-entities/batch", json={"articles": []})
    assert res_empty.status_code == 400


    res_invalid = client.post("/extract-entities/batch", json={"articles": "not a list"})
    assert res_invalid.status_code == 400


def test_chinese_language_extraction():
    payload = {
        "title": "顺应公众反馈，MHC重新安排屠妖节比赛",
        "body": "马来西亚曲棍球联合会确认，在与社区领袖协商后，武吉加里尔的所有MHL比赛将移至明年11月进行。"
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 200
    data = response.json()


    assert data["meta"]["language"] == "zh"
    assert data["meta"]["entity_count"] >= 4

    texts = [ent["text"] for ent in data["entities"]]
    assert "MHC" in texts
    assert "屠妖节" in texts
    assert "马来西亚曲棍球联合会" in texts
    assert "武吉加里尔" in texts


def test_html_tag_stripping():
    payload = {
        "title": "<h1><b>Maybank</b> Expands in <a href='https://example.com'>Kuala Lumpur</a></h1>",
        "body": "<p>Malayan Banking Berhad announced <strong>RM50 million</strong> investments &amp; growth.</p>"
    }
    response = client.post("/extract-entities", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "cleaned_text" in data

    assert "<h1>" not in data["cleaned_text"]
    assert "<b>" not in data["cleaned_text"]
    assert "<p>" not in data["cleaned_text"]
    assert "<strong>" not in data["cleaned_text"]
    assert "Maybank Expands in Kuala Lumpur" in data["cleaned_text"]
    assert "&amp;" not in data["cleaned_text"]
    assert "&" in data["cleaned_text"]


    for ent in data["entities"]:
        assert "<" not in ent["text"]
        assert ">" not in ent["text"]

        assert data["cleaned_text"][ent["start"]:ent["end"]] == ent["text"]


    res_empty_html = client.post("/extract-entities", json={"title": "<p>   <br/> </p>", "body": "Valid body"})
    assert res_empty_html.status_code == 400
