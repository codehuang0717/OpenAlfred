from typing import Optional, Literal
from langchain.tools import ToolRuntime, tool
from langchain.messages import ToolMessage
from langgraph.types import Command
import json

# Local imports
from email_service import get_recent_emails as _get_recent_emails
from email_service import read_email as _read_email
from email_service import draft_and_send_email as _draft_and_send_email
from email_service import EmailServiceException
from tools.todos import _get_user_id

@tool
async def get_recent_emails(
    runtime: ToolRuntime, 
    limit: int = 10, 
    account_id: Optional[str] = None
) -> Command:
    """
    Get a list of recent emails from the user's inbox.
    Returns the subject, sender, date, and email ID for the most recent emails.
    Useful for checking new emails or summarizing recent inbox activity.
    """
    user_id = await _get_user_id(runtime)
    try:
        emails = await _get_recent_emails(user_id=user_id, limit=limit, account_id=account_id)
        # Format for rich rendering in the frontend
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps({
                            "type": "email_list",
                            "emails": emails
                        }),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    except EmailServiceException as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to fetch emails: {str(e)}. Please check your email configuration in settings.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    except Exception as e:
         return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"An unexpected error occurred while fetching emails: {str(e)}.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )


@tool
async def read_email(
    runtime: ToolRuntime, 
    email_id: str, 
    account_id: str
) -> Command:
    """
    Read the full content of a specific email by its ID and Account ID.
    Use this when the user asks to read, summarize, or translate a specific email from the recent emails list.
    You MUST provide both the email_id and the account_id returned from get_recent_emails.
    """
    user_id = await _get_user_id(runtime)
    try:
        email_data = await _read_email(user_id=user_id, email_id=email_id, account_id=account_id)
        
        # Remove html_body so we don't blow up the LLM token limit
        if "html_body" in email_data:
            del email_data["html_body"]
            
        # Truncate body if too long
        if "body" in email_data and len(email_data["body"]) > 4000:
            email_data["body"] = email_data["body"][:4000] + "\n...[Content truncated]..."
            
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps({
                            "type": "email_content",
                            "email": email_data
                        }),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    except EmailServiceException as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to read email {email_id}: {str(e)}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    except Exception as e:
         return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"An unexpected error occurred while reading email: {str(e)}.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )


@tool
async def draft_and_send_email(
    runtime: ToolRuntime, 
    to_address: str, 
    subject: str, 
    body: str,
    account_id: Optional[str] = None
) -> Command:
    """
    Drafts and sends an email on behalf of the user.
    If the user has multiple accounts, you should ask or specify the account_id to send from.
    Always confirm with the user before sending.
    """
    user_id = await _get_user_id(runtime)
    try:
        await _draft_and_send_email(
            user_id=user_id, 
            to=to_address, 
            subject=subject, 
            body=body, 
            account_id=account_id
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Successfully sent email to {to_address} with subject '{subject}'.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    except EmailServiceException as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to send email: {str(e)}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    except Exception as e:
         return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"An unexpected error occurred while sending email: {str(e)}.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )


email_tools = [
    get_recent_emails,
    read_email,
    draft_and_send_email
]
