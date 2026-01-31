# QuantumMind Guideline: Phase 1 - Environment Setup

> Complete Step 1.3 (IBM Token). Steps 1.1, 1.2, 1.4 already done.

---

## Step 1.3: IBM Quantum Token Setup

**Status**: BLOCKED (requires user action)

**Prerequisites**: None

### Actions

1. User must sign up at https://quantum.cloud.ibm.com

2. Get API token from dashboard

3. Set environment variable:
```bash
export IBM_QUANTUM_TOKEN="your_token_here"
```

4. Add to shell profile (~/.bashrc or ~/.zshrc):
```bash
echo 'export IBM_QUANTUM_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

5. Create and run verification script:

```python
# test_ibm_connection.py
from qiskit_ibm_runtime import QiskitRuntimeService
import os

def test_ibm_connection():
    """Verify IBM Quantum connection."""

    # Check token exists
    token = os.environ.get('IBM_QUANTUM_TOKEN')
    if token is None:
        raise EnvironmentError(
            "IBM_QUANTUM_TOKEN not set. "
            "Run: export IBM_QUANTUM_TOKEN='your_token'"
        )
    print(f"Token found: {token[:8]}...{token[-4:]}")

    # Save account
    QiskitRuntimeService.save_account(
        channel="ibm_quantum",
        token=token,
        overwrite=True
    )
    print("Account saved to disk")

    # Connect
    service = QiskitRuntimeService()
    backends = service.backends()
    print(f"\nConnected! Available backends: {len(backends)}")

    # List first 5 backends
    for b in backends[:5]:
        print(f"  - {b.name}: {b.num_qubits} qubits")

    # Verify Eagle processor access
    try:
        eagle = service.backend("ibm_sherbrooke")
        print(f"\nTarget backend: {eagle.name}")
        print(f"  Qubits: {eagle.num_qubits}")
        print(f"  Status: {eagle.status().status_msg}")
        print(f"  Pending jobs: {eagle.status().pending_jobs}")
    except Exception as e:
        print(f"\nWarning: Could not access ibm_sherbrooke: {e}")
        print("May need to check your IBM Quantum plan")
        return False

    print("\nIBM Quantum setup COMPLETE")
    return True


if __name__ == "__main__":
    success = test_ibm_connection()
    exit(0 if success else 1)
```

6. Run: `python test_ibm_connection.py`

7. Verify persistence by running again without setting env var:
```bash
unset IBM_QUANTUM_TOKEN
python test_ibm_connection.py  # Should still work (saved to disk)
```

---

## Verification Checklist

- [ ] IBM Quantum account created
- [ ] API token obtained from dashboard
- [ ] IBM_QUANTUM_TOKEN environment variable set
- [ ] Token added to shell profile for persistence
- [ ] `test_ibm_connection.py` runs without error
- [ ] Shows available backends (should see 5+)
- [ ] Can access ibm_sherbrooke (127 qubits)
- [ ] Token persisted (script works after unsetting env var)

---

## Troubleshooting

**"Token not found"**
- Make sure you exported the variable in the current shell
- Check: `echo $IBM_QUANTUM_TOKEN`

**"Could not access ibm_sherbrooke"**
- Free tier may have limited backend access
- Check IBM Quantum dashboard for your available backends
- Can use any 27+ qubit backend as alternative

**"Connection timeout"**
- Check internet connection
- IBM Quantum may be under maintenance
- Try again in a few minutes

---

## After Completion

Update `00_OVERVIEW.md`:
- Change Phase 1 progress to 4/4
- Mark Step 1.3 as DONE

Next: Proceed to `03_FINETUNING.md` (Step 3.4)

---

**DO NOT PROCEED TO PHASE 3 UNTIL ALL 8 CHECKS PASS**
