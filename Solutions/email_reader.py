import imaplib
import email
import openai # need to be installed via pip
from email.header import decode_header
from datetime import datetime, timedelta
import re

# Increase IMAP fetch line limit to avoid "too many bytes" error
imaplib._MAXLINE = 100000  

# Email credentials
IMAP_SERVER = "imap.gmail.com"  # Adjust for your provider
EMAIL_ACCOUNT = "youremail_for_example_name@gmail.com"
APP_PASSWORD = "gmail_app_password"  # Use an app password for Gmail

# OpenAI API Key
OPENAI_API_KEY = "your_very_long_openai_api_key"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Get the date for filtering (2 days ago)
date_filter = (datetime.today() - timedelta(days=2)).strftime("%d-%b-%Y")

# Allowed email domains
ALLOWED_DOMAINS = {"arizona.edu", "protonmail.com", "proton.me"}

# Only pay attention to emails fomr univeristy of arizona
def extract_domain(email_address):
    """Extract the domain from an email address."""
    match = re.search(r'@([\w.-]+)', email_address)
    return match.group(1).lower() if match else ""

def summarize_text(text):
    """Send email content to ChatGPT for summarization."""
    if not text.strip():
        return "No content to summarize."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following email content."},
                {"role": "user", "content": text}
            ],
            temperature=0.5,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error summarizing email: {e}"

def extract_text_from_email(msg):
    """Extracts and returns the plain text content of an email, skipping attachments."""
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            # Skip attachments (anything with a filename)
            if part.get_filename():
                print(f"Skipping attachment: {part.get_filename()}")
                continue

            # Extract plain text content
            if content_type == "text/plain" and "attachment" not in content_disposition:
                body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore")
                break  # Stop once we get plain text
    else:
        # Non-multipart emails (should be plain text)
        body = msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="ignore")

    return body

# Connect to the Gmail IMAP server
try:
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ACCOUNT, APP_PASSWORD)
    mail.select("INBOX")
except imaplib.IMAP4.error as e:
    print(f"IMAP connection error: {e}")
    exit(1)

# Search for unread emails SINCE the date_filter
search_query = f'(UNSEEN SINCE {date_filter})'
status, messages = mail.search(None, search_query)

email_ids = messages[0].split()

print(f"Found {len(email_ids)} unread emails from the last 2 days.")

# Process emails in batches
batch_size = 50
total_emails = len(email_ids)

for start in range(0, total_emails, batch_size):
    end = min(start + batch_size, total_emails)
    batch_ids = email_ids[start:end]

    print(f"Processing batch {start+1}-{end} of {total_emails} emails...")

    for email_id in batch_ids:
        try:
            # Fetch only the header and text body (skip attachments)
            # status, msg_data = mail.fetch(email_id, "(BODY.PEEK[HEADER] BODY.PEEK[TEXT])")
            status, msg_data = mail.fetch(email_id, "(RFC822)")

            if msg_data is None or len(msg_data) == 0:
                print(f"Skipping email {email_id}: No data returned.")
                continue

            for response_part in msg_data:
                if isinstance(response_part, tuple) and isinstance(response_part[1], bytes):
                    # Parse email
                    msg = email.message_from_bytes(response_part[1])

                    # Extract sender
                    from_header = msg.get("From", "Unknown Sender")
                    sender_domain = extract_domain(from_header)

                    # **Filter emails based on sender domain**
                    if sender_domain not in ALLOWED_DOMAINS:
                        print(f"Skipping email {email_id}: Sender ({from_header}) is not in allowed domains.")
                        continue  # Skip this email

                    # Decode subject
                    subject, encoding = decode_header(msg["Subject"] or "No Subject")[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or "utf-8")

                    # Extract plain text body (skipping attachments)
                    body = extract_text_from_email(msg)

                    if not body.strip():
                        print(f"Skipping email {email_id}: No valid text content found.")
                        continue

                    # Send email body to ChatGPT for summarization
                    summary = summarize_text(body)

                    # Print the summarized email
                    print(f"\nEmail Subject: {subject}")
                    print(f"From: {from_header}")
                    print(f"Summary: {summary}")

        except imaplib.IMAP4.error as e:
            print(f"Error fetching email {email_id}: {e}")
            continue  # Skip to the next email if there's an error

# Close connection
mail.logout()
print("Finished processing emails.")