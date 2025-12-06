import json
import requests
import sys
from pprint import pprint

SERVER_URL = "http://127.0.0.1:8000"
MAX_QUESTIONS = 5
TOP_K_OUTPUT = 3

# FIXED: Matching server structure
TRUSTED_SITES = [
    {"url": "https://www.cdc.gov/health/conditions", "category": "Government Health"},
    {"url": "https://www.who.int/health-topics", "category": "International Health"},
    {"url": "https://www.nhs.uk/conditions", "category": "UK Health Service"},
    {"url": "https://www.mayoclinic.org/diseases-conditions", "category": "Medical Research"},
    {"url": "https://www.medicinenet.com/conditions.htm", "category": "Medical Reference"},
    {"url": "https://www.webmd.com/a-to-z-guides/health-topics", "category": "Health Information"},
    {"url": "https://www.healthline.com/health", "category": "Health Information"},
    {"url": "https://www.nih.gov/health-information", "category": "Government Health"},
    {"url": "https://www.clevelandclinic.org/health/diseases", "category": "Medical Research"},
    {"url": "https://medlineplus.gov/encyclopedia.html", "category": "Medical Encyclopedia"},
]


def ingest_url_interactive():
    url = input("URL to ingest:\n> ").strip()
    if not url:
        print("No URL provided.")
        return
    resp = requests.post(SERVER_URL + "/ingest_url", json={"url": url})
    if resp.status_code != 200:
        print("Error ingesting:", resp.text)
        return
    print("Ingest result:", resp.json())


def ingest_medical_urls():
    print("🔄 Ingesting trusted medical websites into RAG index...")
    success = 0
    failed = 0
    for site in TRUSTED_SITES:
        url = site["url"]
        try:
            resp = requests.post(SERVER_URL + "/ingest_url", json={"url": url}, timeout=30)
            if resp.status_code == 200:
                print(f"✅ Ingested: {url}")
                success += 1
            else:
                print(f"❌ Failed to ingest: {url} - {resp.text}")
                failed += 1
        except Exception as e:
            print(f"❌ Error ingesting: {url} - {str(e)}")
            failed += 1

    print(f"\n📊 Ingestion complete: {success} successful, {failed} failed")

    # After ingestion, build the index
    if success > 0:
        build_index()


def build_index():
    resp = requests.post(SERVER_URL + "/build_index", json={"persist": True})
    if resp.status_code != 200:
        print("Error building index:", resp.text)
        return
    print("Build result:", resp.json())


def index_info():
    resp = requests.get(SERVER_URL + "/index_info")
    info = resp.json()
    print(f"\n📊 RAG Index Info:")
    print(f"  - Chunks: {info.get('num_chunks', 0)}")
    print(f"  - Dimension: {info.get('dim', 'N/A')}")
    print(f"  - FAISS enabled: {info.get('has_faiss', False)}")

    samples = info.get('sample_content', [])
    if samples:
        print(f"\n📄 Sample content in index:")
        for i, sample in enumerate(samples, 1):
            print(f"\n  {i}. Category: {sample.get('category', 'Unknown')}")
            print(f"     Source: {sample.get('source', 'Unknown')}")
            print(f"     Preview: {sample.get('text_preview', '')[:100]}...")
    else:
        print("\n⚠️  No content in index!")


def serve_loop():
    # Check if index is empty
    resp = requests.get(SERVER_URL + "/index_info")
    index_data = resp.json()

    print("\n" + "=" * 60)
    print("🏥 MEDICAL SYMPTOM TRIAGE (RAG-ENABLED)")
    print("=" * 60)

    if index_data.get("num_chunks", 0) == 0:
        print("\n⚠️  No RAG index found!")
        choice = input("Would you like to ingest trusted medical websites now? (y/n): ").strip().lower()
        if choice == 'y':
            ingest_medical_urls()
        else:
            print("⚠️  Proceeding without RAG - diagnoses may be less accurate")
    else:
        print(f"✅ RAG index loaded with {index_data['num_chunks']} knowledge chunks")

    print("\nCommands: /ingest  /build  /index  (or describe symptoms to start)")
    user_text = input("\nDescribe your symptoms (or command):\n> ").strip()

    if user_text.startswith("/ingest"):
        ingest_url_interactive()
        return
    if user_text.startswith("/build"):
        build_index()
        return
    if user_text.startswith("/index"):
        index_info()
        return

    conversation_history = f"Patient initial complaint: {user_text}"
    last_ai_content = None

    for qn in range(MAX_QUESTIONS):
        print(f"\n🔄 Analyzing symptoms (Round {qn + 1}/{MAX_QUESTIONS})...")
        payload = {"history": conversation_history, "max_questions": 1}

        try:
            r = requests.post(SERVER_URL + "/triage_step", json=payload, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"❌ Server error: {str(e)}")
            return

        try:
            data = r.json()
        except Exception:
            print("⚠️ Warning: Server returned invalid JSON. Skipping this round.")
            break

        last_ai_content = data

        # Display candidates
        candidates = data.get("candidates", [])
        if not candidates:
            print("⚠️ No conditions returned by the server.")
            candidates = []

        candidates = sorted(candidates, key=lambda x: x.get("probability", 0), reverse=True)

        # FIXED: Show if RAG evidence was used
        evidence_used = data.get("evidence_used", False)
        evidence_available = data.get("_evidence_available", False)
        evidence_reasoning = data.get("evidence_reasoning", "")

        print(f"\n{'📚 RAG EVIDENCE USED ✅' if evidence_used else '⚠️  NO RAG EVIDENCE USED'}")
        if evidence_reasoning:
            print(f"   Reasoning: {evidence_reasoning}")
        elif evidence_available and not evidence_used:
            print("   (Evidence was available but model didn't use it effectively)")

        print("\n🔍 Top suspected conditions:")
        for idx, item in enumerate(candidates[:TOP_K_OUTPUT], 1):
            condition = item.get('condition', 'Unknown')
            prob = float(item.get('probability', 0)) * 100
            print(f"  {idx}. {condition:<35} {prob:>5.1f}%")

        # Show retrieved sources if available
        retrieved = data.get("_retrieved", [])
        if retrieved:
            print(f"\n📖 Retrieved {len(retrieved)} sources:")
            for idx, r in enumerate(retrieved[:3], 1):
                score = r.get("score", 0)
                title = r.get("source_title", "Unknown")[:50]
                text_preview = r.get("text", "")[:150]
                print(f"  {idx}. [{score:.2f}] {title}")
                print(f"     Preview: {text_preview}...")
        else:
            print(f"\n⚠️  No sources retrieved")

        # Check next question
        question = data.get("next_question", "DIAGNOSIS_COMPLETE")
        if question == "DIAGNOSIS_COMPLETE":
            print("\n✅ AI indicates diagnosis complete.")
            break

        print(f"\n❓ Follow-up question: {question}")
        ans = input("Your answer (or 'report' to finish):\n> ").strip()

        if ans.lower() in ['report', 'done', 'stop', 'exit', 'quit', 'q']:
            print("\n⏹️  Stopping interview and generating report...")
            break

        conversation_history += f" | Q: {question} A: {ans}"

    # Final report
    print("\n" + "=" * 60)
    print("📋 FINAL MEDICAL ANALYSIS")
    print("=" * 60)

    if last_ai_content:
        candidates = last_ai_content.get("candidates", [])
        if not candidates:
            print("⚠️ No conditions returned in final report.")
            candidates = []

        candidates = sorted(candidates, key=lambda x: x.get("probability", 0), reverse=True)

        print(f"\n{'CONDITION':<40} | {'CONFIDENCE':>10}")
        print("-" * 55)
        for item in candidates[:TOP_K_OUTPUT]:
            condition = item.get('condition', 'Unknown')
            prob = int(float(item.get('probability', 0)) * 100)
            print(f"{condition:<40} | {prob:>10}%")

        print("\n" + "=" * 60)
        print("💡 RECOMMENDATION:")
        print("=" * 60)
        recommendation = last_ai_content.get("top_recommendation", "No recommendation returned.")
        print(recommendation)

        # Show sources used
        retrieved = last_ai_content.get("_retrieved", [])
        if retrieved:
            print("\n" + "=" * 60)
            print("📚 MEDICAL SOURCES REFERENCED:")
            print("=" * 60)
            for idx, r in enumerate(retrieved[:5], 1):
                score = r.get("score", 0)
                url = r.get("url", "Unknown")
                title = r.get("source_title", "Unknown")
                print(f"{idx}. [{score:.2f}] {title}")
                print(f"   {url}")
        else:
            print("\n⚠️ No medical sources were referenced in this diagnosis")

        # RAG usage summary
        evidence_used = last_ai_content.get("evidence_used", False)
        evidence_reasoning = last_ai_content.get("evidence_reasoning", "")
        print("\n" + "=" * 60)
        if evidence_used:
            print("✅ This diagnosis WAS informed by trusted medical sources")
            if evidence_reasoning:
                print(f"\n   {evidence_reasoning}")
        else:
            print("⚠️  WARNING: This diagnosis was NOT based on RAG evidence")
            if evidence_reasoning:
                print(f"\n   Reason: {evidence_reasoning}")
            else:
                print("   Consider re-running with proper RAG index loaded")
        print("=" * 60)
    else:
        print("❌ No analysis data available.")

    print("\n💬 Disclaimer: This is an AI triage tool, not a replacement for")
    print("   professional medical advice. Always consult a healthcare provider.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "ingest":
            ingest_url_interactive()
        elif cmd == "build":
            build_index()
        elif cmd == "index":
            index_info()
        else:
            print("Unknown command")
    else:
        serve_loop()