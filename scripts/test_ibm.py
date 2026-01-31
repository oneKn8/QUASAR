"""
IBM Quantum connection test script.

This script verifies the connection to IBM Quantum services.
Run with: python scripts/test_ibm.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_ibm_connection():
    """Test connection to IBM Quantum."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("IBM_QUANTUM_TOKEN")

    if not token:
        print("WARNING: IBM_QUANTUM_TOKEN not found in environment.")
        print("To use IBM Quantum hardware, set your token:")
        print("  export IBM_QUANTUM_TOKEN='your_token_here'")
        print("")
        print("Get your token from: https://quantum.cloud.ibm.com")
        print("")
        print("Checking if account is already saved...")

        try:
            service = QiskitRuntimeService()
            print("SUCCESS: Found saved IBM Quantum account.")
        except Exception as e:
            print(f"No saved account found: {e}")
            print("\nTo save your account, run:")
            print("  from qiskit_ibm_runtime import QiskitRuntimeService")
            print("  QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')")
            return False
    else:
        print(f"Found IBM_QUANTUM_TOKEN in environment.")
        try:
            # Save account if not already saved
            QiskitRuntimeService.save_account(
                channel="ibm_quantum",
                token=token,
                overwrite=True
            )
            print("Account saved successfully.")
            service = QiskitRuntimeService()
        except Exception as e:
            print(f"Error saving account: {e}")
            return False

    # List available backends
    try:
        backends = service.backends()
        print(f"\nAvailable backends: {len(backends)}")

        # Filter for simulators and real hardware
        simulators = [b for b in backends if b.simulator]
        real_hw = [b for b in backends if not b.simulator]

        print(f"  Simulators: {len(simulators)}")
        print(f"  Real hardware: {len(real_hw)}")

        if real_hw:
            print("\nReal quantum hardware:")
            for b in real_hw[:5]:
                status = b.status()
                print(f"  - {b.name}: {b.num_qubits} qubits, status: {status.status_msg}")

        return True

    except Exception as e:
        print(f"Error listing backends: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("IBM Quantum Connection Test")
    print("=" * 50)
    print("")

    success = test_ibm_connection()

    print("")
    print("=" * 50)
    if success:
        print("IBM Quantum connection: SUCCESS")
    else:
        print("IBM Quantum connection: FAILED (see above for details)")
    print("=" * 50)

    sys.exit(0 if success else 1)
