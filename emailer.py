# ------------------------------------------------------------------------------------
# emailer.py:
# Helper for sending emails via SendGrid API.
# ------------------------------------------------------------------------------------
from typing import List, Optional
import requests, json
from Config import SENDGRID_API_KEY, EMAIL_FROM, SENDGRID_TEMPLATE_ID

def send_email(to: List[str], subject: str, html: str, plain: Optional[str] = None):
    """
    Send an email through the SendGrid API using a dynamic template.

    Args:
        to (List[str]): List of recipient email addresses.
        subject (str): Subject line of the email.
        html (str): HTML content (injected into the template as 'summary_html').
        plain (Optional[str]): Optional plaintext fallback (defaults to empty string).

    Raises:
        RuntimeError: If SendGrid returns an error (HTTP 300+).
    """
    # SendGrid API endpoint for sending mail
    url = "https://api.sendgrid.com/v3/mail/send"

    # Auth + content headers
    headers = {
        "Authorization": f"Bearer {SENDGRID_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build request payload:
    # "from": must be a verified sender in SendGrid
    # "personalizations": recipients and template variables
    # "template_id": ID of the dynamic template configured in SendGrid
    data = {
        "from": {"email": EMAIL_FROM},
        "personalizations": [{
            "to": [{"email": x} for x in to],
            "dynamic_template_data": {
                "subject": subject,
                "summary_html": html,   
                "summary_text": plain or "",
            }
        }],
        "template_id": SENDGRID_TEMPLATE_ID
    }

    # Perform the HTTP POST request to SendGrid
    r = requests.post(url, headers=headers, data=json.dumps(data), timeout=20)

    # Raise an error if SendGrid didn’t accept the request
    if r.status_code >= 300:
        raise RuntimeError(f"SendGrid error {r.status_code}: {r.text[:300]}")
