from typing import Optional, Literal
from langchain.tools import ToolRuntime, tool
from langchain.messages import ToolMessage
from langgraph.types import Command
import json

# Local imports
from services.email import get_recent_emails as _get_recent_emails
from services.email import read_email as _read_email
from services.email import draft_and_send_email as _draft_and_send_email
from services.email import EmailServiceException
from tools.todos import _get_user_id

@tool
async def get_recent_emails(
    runtime: ToolRuntime, 
    limit: int = 10, 
    account_filter: Optional[str] = None
) -> Command:
    """Get recent emails from user's inbox. 
    If the user specifies an email provider or address (e.g., 'qq', 'gmail'), pass it as account_filter to strictly fetch from that account.
    Returns subject, sender, date, email ID, and available account info."""
    user_id = await _get_user_id(runtime)
    try:
        from core.database import get_email_credentials
        creds = await get_email_credentials(user_id)
        accounts = [{"account_id": c["account_id"], "email": c["email_address"], "provider": c["provider"]} for c in creds]
        
        target_account_ids = None
        if account_filter:
            filter_lower = account_filter.lower()
            target_account_ids = [
                c["account_id"] for c in accounts 
                if filter_lower in c["email"].lower() or filter_lower in c["provider"].lower()
            ]
            if not target_account_ids:
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=f"No email accounts found matching '{account_filter}'. Available accounts: {json.dumps(accounts)}",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
                
        emails = await _get_recent_emails(user_id=user_id, limit=limit, account_ids=target_account_ids)
        # Format for rich rendering in the frontend
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps({
                            "type": "email_list",
                            "emails": emails,
                            "accounts": accounts
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
    """Read the full content of a specific email. Requires both email_id and account_id from get_recent_emails."""
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


email_tools = [
    get_recent_emails,
    read_email,
]
