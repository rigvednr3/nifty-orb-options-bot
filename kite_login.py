"""
Run once every trading morning (Kite access tokens expire daily, ~6 AM).

    export KITE_API_KEY=xxxx KITE_API_SECRET=yyyy
    python kite_login.py

Open the printed URL, log in, copy the `request_token` from the redirect URL, paste it here.
"""
import os

from kiteconnect import KiteConnect

k = KiteConnect(api_key=os.environ["KITE_API_KEY"])
print("Login here:\n", k.login_url())
req = input("\nPaste request_token: ").strip()
sess = k.generate_session(req, api_secret=os.environ["KITE_API_SECRET"])
path = os.environ.get("KITE_TOKEN_FILE", ".kite_token")
with open(path, "w") as f:
    f.write(sess["access_token"])
os.chmod(path, 0o600)
print(f"Access token saved to {path}")
