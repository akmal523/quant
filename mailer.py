import smtplib
from email.message import EmailMessage
import os

def send_results_email(
    sender_email: str, 
    sender_password: str, 
    recipient_email: str, 
    output_dir: str = "outputs"
) -> None:
    msg = EmailMessage()
    msg['Subject'] = "Quant-AI v6 Sector Scan & Portfolio Audit"
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg.set_content("Automated quantitative scan and portfolio audit attached.")

    files_to_attach = ["market_scan.xlsx", "market_scan.csv", "portfolio_audit.csv"]

    for filename in files_to_attach:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                file_data = f.read()
                msg.add_attachment(
                    file_data, 
                    maintype='application', 
                    subtype='octet-stream', 
                    filename=filename
                )

    # Standard TLS configuration for common SMTP (e.g., Gmail: smtp.gmail.com)
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print("[Mailer] Transmission successful.")
    except Exception as e:
        print(f"[Mailer] Transmission failed: {e}")

if __name__ == "__main__":
    # Execute standalone or import into main.py phase 5
    send_results_email(
        sender_email="your_email@gmail.com",
        sender_password="your_app_password",
        recipient_email="target_email@domain.com"
    )
