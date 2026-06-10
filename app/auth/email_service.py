"""Transactional email via Gmail SMTP — OTP verification and password-reset links.

Uses Python's built-in smtplib (no extra packages needed).
Configured via SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD in .env.
"""
from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_FROM = f"{settings.EMAIL_FROM_NAME} <{settings.EMAIL_FROM}>"


def _send(to_email: str, subject: str, html: str) -> None:
    """Send a single HTML email via SMTP. Raises RuntimeError if not configured."""
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        raise RuntimeError(
            "SMTP_USER / SMTP_PASSWORD not set. Add them to your .env file."
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = _FROM
    msg["To"]      = to_email
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        server.sendmail(settings.EMAIL_FROM, to_email, msg.as_string())


def send_otp_email(to_email: str, otp_code: str) -> None:
    """Send a 6-digit OTP to the user's email address."""
    if settings.DEV_OTP_LOG:
        logger.warning(event="dev_otp_log", to=to_email, otp=otp_code,
                       note="DEV_OTP_LOG=true — not sent via email")
        return
    html = f"""
<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#0a0a0a;font-family:'Inter',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0a0a0a;padding:40px 0;">
    <tr><td align="center">
      <table width="480" cellpadding="0" cellspacing="0"
        style="background:#141414;border:1px solid #2a2a2a;border-radius:16px;overflow:hidden;">

        <tr>
          <td style="background:linear-gradient(135deg,#8b5cf6,#3b82f6);padding:28px 40px;text-align:center;">
            <h1 style="margin:0;color:#fff;font-size:22px;font-weight:700;letter-spacing:-0.3px;">
              {settings.EMAIL_FROM_NAME}
            </h1>
            <p style="margin:6px 0 0;color:rgba(255,255,255,0.75);font-size:13px;">
              Sign-in verification code
            </p>
          </td>
        </tr>

        <tr>
          <td style="padding:36px 40px;text-align:center;">
            <p style="margin:0 0 24px;color:#c0c0c0;font-size:15px;line-height:1.6;">
              Enter this code to complete your sign-in.
              It expires in <strong style="color:#e5e5e5;">10 minutes</strong>.
            </p>
            <div style="display:inline-block;background:#1c1c1c;border:1px solid #404040;
                        border-radius:12px;padding:20px 48px;margin:0 0 24px;">
              <span style="font-size:36px;font-weight:800;letter-spacing:10px;
                           color:#fff;font-family:'Courier New',monospace;">
                {otp_code}
              </span>
            </div>
            <p style="margin:0;color:#6e6e6e;font-size:13px;line-height:1.6;">
              If you didn't try to sign in, you can safely ignore this email.
            </p>
          </td>
        </tr>

        <tr>
          <td style="border-top:1px solid #1a1a1a;padding:20px 40px;text-align:center;">
            <p style="margin:0;color:#6e6e6e;font-size:12px;">
              &copy; {settings.EMAIL_FROM_NAME} &middot; This is an automated message, please do not reply.
            </p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
"""
    try:
        _send(to_email, f"Your {settings.EMAIL_FROM_NAME} sign-in code: {otp_code}", html)
        logger.info(event="otp_email_sent", to=to_email)
    except Exception as exc:
        logger.error(event="otp_email_failed", to=to_email, error=str(exc))
        raise


def send_password_reset_email(to_email: str, reset_token: str) -> None:
    """Send a password-reset link containing the one-time reset token."""
    reset_url = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
    if settings.DEV_OTP_LOG:
        logger.warning(event="dev_reset_log", to=to_email, reset_url=reset_url,
                       note="DEV_OTP_LOG=true — not sent via email")
        return

    html = f"""
<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#0a0a0a;font-family:'Inter',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0a0a0a;padding:40px 0;">
    <tr><td align="center">
      <table width="480" cellpadding="0" cellspacing="0"
        style="background:#141414;border:1px solid #2a2a2a;border-radius:16px;overflow:hidden;">

        <tr>
          <td style="background:linear-gradient(135deg,#8b5cf6,#3b82f6);padding:28px 40px;text-align:center;">
            <h1 style="margin:0;color:#fff;font-size:22px;font-weight:700;letter-spacing:-0.3px;">
              {settings.EMAIL_FROM_NAME}
            </h1>
            <p style="margin:6px 0 0;color:rgba(255,255,255,0.75);font-size:13px;">
              Password reset request
            </p>
          </td>
        </tr>

        <tr>
          <td style="padding:36px 40px;text-align:center;">
            <p style="margin:0 0 28px;color:#c0c0c0;font-size:15px;line-height:1.6;">
              We received a request to reset the password for
              <strong style="color:#e5e5e5;">{to_email}</strong>.
              This link expires in <strong style="color:#e5e5e5;">1 hour</strong>.
            </p>
            <a href="{reset_url}"
               style="display:inline-block;background:linear-gradient(135deg,#8b5cf6,#3b82f6);
                      color:#fff;text-decoration:none;font-size:15px;font-weight:600;
                      padding:14px 36px;border-radius:12px;letter-spacing:0.2px;">
              Reset my password
            </a>
            <p style="margin:28px 0 0;color:#6e6e6e;font-size:13px;line-height:1.6;">
              If the button doesn't work, copy and paste this link:<br/>
              <span style="color:#8b5cf6;word-break:break-all;">{reset_url}</span>
            </p>
            <p style="margin:20px 0 0;color:#6e6e6e;font-size:13px;">
              If you didn't request this, you can safely ignore this email.
            </p>
          </td>
        </tr>

        <tr>
          <td style="border-top:1px solid #1a1a1a;padding:20px 40px;text-align:center;">
            <p style="margin:0;color:#6e6e6e;font-size:12px;">
              &copy; {settings.EMAIL_FROM_NAME} &middot; This is an automated message, please do not reply.
            </p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
"""
    try:
        _send(to_email, f"Reset your {settings.EMAIL_FROM_NAME} password", html)
        logger.info(event="reset_email_sent", to=to_email)
    except Exception as exc:
        logger.error(event="reset_email_failed", to=to_email, error=str(exc))
        raise
