# Email Setup for Continuous Learning System

The continuous learning orchestrator sends automated emails for:
- **Weekly reports**: Performance summaries every Monday at 9 AM
- **Alerts**: Immediate notifications for drift, performance drops, storage issues, and errors

---

## Gmail Setup (Recommended)

### 1. Enable 2-Factor Authentication

1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Enable **2-Step Verification** if not already enabled

### 2. Generate App Password

1. Go to [App Passwords](https://myaccount.google.com/apppasswords)
2. Select app: **Mail**
3. Select device: **Other (Custom name)**
4. Enter name: **Continuous Learning System**
5. Click **Generate**
6. Copy the 16-character password (e.g., `abcd efgh ijkl mnop`)


### 3. Configure Email Password

The email password is stored in the `../.env` file (parent directory of XOR project) for security.

**Add to `../.env` file:**
```bash
# Email password for continuous learning alerts (Gmail app password)
EMAIL_PASSWORD="abcdefghijklmnop"  # Remove spaces from app password
```

The system will automatically load the password from this file. This is more secure than shell environment variables in `.bashrc`/`.zshrc` which may be exposed in process listings.

**Priority order for password loading:**

1. `../.env` file (recommended)
2. Config file `email_password` field
3. Environment variable `EMAIL_PASSWORD`

### 4. Update Configuration File

Edit [`config/continuous_learning_config.yaml`](../config/continuous_learning_config.yaml):

```yaml
orchestration:
  email_recipients:
    - your-email@gmail.com  # Your actual Gmail address
  email_smtp_server: smtp.gmail.com
  email_smtp_port: 587
  email_sender: your-email@gmail.com  # Same as sender
  email_password: null  # Will be read from EMAIL_PASSWORD env var
```

### 5. Test Email Sending

```python
import os
from continuous.orchestrator import EmailReporter

# Mock config for testing
class MockOrchestrationConfig:
    email_recipients = ['recipient@example.com']
    email_smtp_server = 'smtp.gmail.com'
    email_smtp_port = 587
    email_sender = 'your-email@gmail.com'
    email_password = os.getenv('EMAIL_PASSWORD')

class MockConfig:
    orchestration = MockOrchestrationConfig()

# Test email
reporter = EmailReporter(MockConfig())
success = reporter.send_email(
    subject="Test Email from Continuous Learning",
    body="This is a test email. If you receive this, email setup is working!"
)

if success:
    print("✓ Email sent successfully!")
else:
    print("✗ Email failed. Check logs for details.")
```

---

## Alternative SMTP Providers

### Outlook/Hotmail

```yaml
orchestration:
  email_smtp_server: smtp-mail.outlook.com
  email_smtp_port: 587
  email_sender: your-email@outlook.com
```

**App Password**: [Create app password](https://account.microsoft.com/security)

### Yahoo Mail

```yaml
orchestration:
  email_smtp_server: smtp.mail.yahoo.com
  email_smtp_port: 587
  email_sender: your-email@yahoo.com
```

**App Password**: [Generate app password](https://login.yahoo.com/account/security)

### SendGrid (Production Recommended)

For production systems, consider using SendGrid for better deliverability:

```yaml
orchestration:
  email_smtp_server: smtp.sendgrid.net
  email_smtp_port: 587
  email_sender: noreply@yourdomain.com
  # Use SendGrid API key as password
```

1. Sign up at [SendGrid](https://sendgrid.com)
2. Create API key with Mail Send permissions
3. Set `EMAIL_PASSWORD` to the API key

---

## Security Best Practices

### ✅ DO

1. **Use environment variables** for passwords (never commit to git)
2. **Use app-specific passwords** (not your main account password)
3. **Limit email recipients** to authorized personnel only
4. **Use dedicated email** for system notifications (not personal email)
5. **Enable 2FA** on your email account
6. **Rotate passwords** regularly (every 6 months)

### ❌ DON'T

1. **Don't commit passwords** to version control
2. **Don't use your main password** (always use app passwords)
3. **Don't share email credentials** across multiple systems
4. **Don't disable TLS/SSL** (always use port 587 or 465)
5. **Don't send sensitive data** in email bodies (use attachments if needed)

---

## Troubleshooting

### "Authentication failed" Error

**Problem**: Email password is incorrect or not set.

**Solutions**:

1. Check `../.env` file contains `EMAIL_PASSWORD="your-app-password"`
2. Verify app password is correct (no spaces)
3. Ensure 2FA is enabled and app password is generated
4. Try regenerating the app password

### "Connection refused" Error

**Problem**: SMTP server or port is incorrect.

**Solutions**:
1. Verify SMTP server: `nslookup smtp.gmail.com`
2. Check port (587 for TLS, 465 for SSL)
3. Ensure firewall allows outbound SMTP
4. Try alternative port (587 ↔ 465)

### "Sender address rejected" Error

**Problem**: Sender email doesn't match authenticated account.

**Solutions**:
1. Ensure `email_sender` matches your Gmail address
2. For custom domains, verify domain ownership
3. Check SPF/DKIM records for custom domains

### Emails Going to Spam

**Problem**: Emails marked as spam by recipient.

**Solutions**:
1. Add sender to safe senders list
2. Check email content (avoid spam trigger words)
3. Use dedicated SMTP service (SendGrid, Mailgun)
4. Configure SPF/DKIM/DMARC for custom domains

### No Emails Received

**Problem**: Emails sent but not received.

**Solutions**:
1. Check spam/junk folder
2. Verify recipient email address is correct
3. Check email logs: `grep "Email sent" logs/experiment.log`
4. Test with simple email first (see Test Email Sending above)

---

## Email Templates

The system sends two types of emails:

### 1. Weekly Report Email

**Subject**: `Continuous Learning Weekly Report - YYYY-MM-DD`

**Content**:
- Performance metrics (accuracy, ROC AUC, precision, recall)
- Data collection summary (files, samples, duration)
- Model updates performed
- Long-term trends (12-week chart)
- Data drift analysis
- Alerts (if any)

**Frequency**: Weekly (configurable day/time)

### 2. Alert Email

**Subject**: `⚠️ Continuous Learning Alert: DRIFT` (or PERFORMANCE, STORAGE, ERROR)

**Content**:
- Alert type and severity
- Timestamp
- Detailed metrics
- Recommended actions

**Frequency**: Immediate (when condition detected)

---

## Disabling Email Alerts

To run without email (for testing):

### Option 1: Empty Recipients List

```yaml
orchestration:
  email_recipients: []  # No emails will be sent
```

### Option 2: Disable Alerts

```yaml
orchestration:
  alert_on_drift: false
  alert_on_performance_drop: false
  # Weekly reports will still be generated but not emailed
```

### Option 3: Dry Run Mode

```bash
python -m src.continuous.orchestrator \
    --config config/continuous_learning_config.yaml \
    --db test.db \
    --dry-run  # Skips all emails
```

---

## Monitoring Email Delivery

### Check Logs

```bash
# View email-related log entries
grep -i "email" logs/experiment.log

# Check for errors
grep -i "email.*error" logs/experiment.log

# View sent emails
grep "Email sent:" logs/experiment.log
```

### Email Delivery Metrics

Track email delivery success rate:

```python
# In monitoring dashboard
from continuous.monitoring_reporter import SystemMonitor

monitor = SystemMonitor(db_path)
health = monitor.get_system_health()

# Add email tracking
email_success_rate = emails_sent / emails_attempted
```

---

## Advanced Configuration

### Custom Email Templates

To customize email content, edit [`src/continuous/orchestrator.py`](../src/continuous/orchestrator.py):

```python
def _markdown_to_html(markdown: str) -> str:
    """Simple markdown to HTML conversion"""
    # Add custom CSS styling
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h1 {{ color: #333; }}
            .alert {{ background-color: #ffcccc; padding: 10px; }}
        </style>
    </head>
    <body>
        {markdown_converted}
    </body>
    </html>
    """
    return html
```

### Multiple Recipient Groups

```yaml
orchestration:
  email_recipients:
    - team-lead@example.com
    - ml-engineer@example.com
    - devops@example.com
    - monitoring@example.com
```

### Rate Limiting

To avoid spam filters, limit alert frequency:

```python
# In orchestrator.py
class EmailReporter:
    def __init__(self, config):
        self.last_alert_time = {}
        self.min_alert_interval = timedelta(hours=1)  # Max 1 alert/hour per type

    def send_alert(self, alert_type, details):
        # Rate limit alerts
        if alert_type in self.last_alert_time:
            time_since_last = datetime.now() - self.last_alert_time[alert_type]
            if time_since_last < self.min_alert_interval:
                logger.info(f"Alert rate-limited: {alert_type}")
                return False

        # Send alert
        success = self.send_email(...)
        if success:
            self.last_alert_time[alert_type] = datetime.now()
        return success
```

---

## Summary

✅ **Required Steps**:

1. Enable 2FA on email account
2. Generate app-specific password
3. Add `EMAIL_PASSWORD="your-password"` to `../.env` file
4. Update `email_recipients` and `email_sender` in config
5. Test email sending

⚠️ **Important**:

- Store password in `../.env` file (outside project directory)
- Never commit email passwords to git
- Use app passwords, not main account passwords
- Start with test emails before production
- Monitor email logs for delivery issues

📧 **Support**:
If issues persist, check the troubleshooting section or review logs at `logs/experiment.log`.
