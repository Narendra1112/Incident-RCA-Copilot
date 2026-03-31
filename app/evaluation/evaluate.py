import requests
import json

API_URL = "http://127.0.0.1:8000/analyze"


TEST_CASES = TEST_CASES = [
    {
        "name": "latency_spike_generic",
        "query": "Why did inference latency spike?",
        "expect_refusal": None,
    },
    {
        "name": "latency_spike_feature_fetcher",
        "query": "Why did feature fetch latency increase in inference API?",
        "expect_refusal": False,
    },
    {
        "name": "worker_saturation",
        "query": "Why were requests rejected due to inflight limit and queue wait time increased?",
        "expect_refusal": False,
    },
    {
        "name": "timeout_issue",
        "query": "Why are requests timing out in inference API?",
        "expect_refusal": False,
    },
    {
        "name": "cache_issue",
        "query": "Why did cache miss ratio increase in feature store?",
        "expect_refusal": False,
    },
    {
        "name": "retry_storm",
        "query": "Why are repeated retries happening in request router?",
        "expect_refusal": False,
    },
    {
        "name": "invalid_query_gpu",
        "query": "Why did GPU memory overflow in training cluster?",
        "expect_refusal": True,
    },
]


REQUIRED_KEYS = [
    "issue_type",
    "likely_root_cause",
    "supporting_evidence",
    "confidence",
    "recommended_next_actions",
    "insufficient_data_flag",
    "stale_evidence_warning",
]


def validate_response_structure(resp: dict) -> bool:
    for key in REQUIRED_KEYS:
        if key not in resp:
            return False
    return True


def validate_supporting_evidence(resp: dict) -> bool:
    evidence = resp.get("supporting_evidence")

    if resp.get("insufficient_data_flag"):
        return True  # allowed empty

    return isinstance(evidence, list) and len(evidence) > 0


def run_test(test):
    try:
        response = requests.post(API_URL, json={"question": test["query"]})
        data = response.json()

        structure_ok = validate_response_structure(data)
        evidence_ok = validate_supporting_evidence(data)

        expected_refusal = test["expect_refusal"]
        if expected_refusal is None:
            refusal_ok = True
        else:
            refusal_ok = (data["insufficient_data_flag"] == expected_refusal)
        success = structure_ok and evidence_ok and refusal_ok

        return {
            "name": test["name"],
            "success": success,
            "structure": structure_ok,
            "evidence": evidence_ok,
            "refusal": refusal_ok,
            "confidence": data.get("confidence"),
        }

    except Exception as e:
        return {
            "name": test["name"],
            "success": False,
            "error": str(e),
        }


def main():
    print("\nRunning Evaluation...\n")

    results = []
    for test in TEST_CASES:
        result = run_test(test)
        results.append(result)

        print(f"Test: {result['name']}")
        print(f"  Success: {result['success']}")

        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Structure OK: {result['structure']}")
            print(f"  Evidence OK: {result['evidence']}")
            print(f"  Refusal OK: {result['refusal']}")
            print(f"  Confidence: {result['confidence']}")

        print("-" * 40)

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"\nFinal Score: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()