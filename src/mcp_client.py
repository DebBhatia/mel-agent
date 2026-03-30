"""
MCP CLIENT — Model Context Protocol (stdio transport)
======================================================
Lightweight async client for communicating with MCP servers via stdin/stdout.
Supports tool discovery, tool invocation, and graceful lifecycle management.

Usage:
    client = MCPClient("npx", ["-y", "@anthropic/google-calendar-mcp"])
    await client.start()
    tools = await client.list_tools()
    result = await client.call_tool("list_events", {"days": 7})
    await client.stop()
"""

import os
import json
import asyncio
import logging
from typing import Optional

logger = logging.getLogger("mcp_client")


class MCPClient:
    """Async JSON-RPC client for MCP servers over stdio transport."""

    def __init__(self, command: str, args: list = None, env: dict = None):
        self.command = command
        self.args = args or []
        self.env = {**os.environ, **(env or {})}
        self._process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._tools: list[dict] = []
        self._initialized = False

    async def start(self) -> bool:
        """Start the MCP server subprocess and initialize the session."""
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.command, *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env,
            )
            self._reader_task = asyncio.create_task(self._read_loop())

            # MCP initialize handshake
            result = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mel-agent", "version": "1.0.0"},
            })
            if result is None:
                logger.error("MCP initialize failed")
                return False

            # Send initialized notification (no response expected)
            await self._send_notification("notifications/initialized", {})
            self._initialized = True
            logger.info(f"MCP server started: {self.command} {' '.join(self.args)}")
            return True
        except FileNotFoundError:
            logger.error(f"MCP server not found: {self.command}")
            return False
        except Exception as e:
            logger.error(f"MCP start error: {e}")
            return False

    async def stop(self):
        """Gracefully shut down the MCP server."""
        if self._process and self._process.returncode is None:
            try:
                self._process.stdin.close()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
            except Exception:
                pass
        if self._reader_task:
            self._reader_task.cancel()
        self._initialized = False

    async def list_tools(self) -> list[dict]:
        """Discover available tools from the MCP server."""
        if not self._initialized:
            return []
        result = await self._send_request("tools/list", {})
        self._tools = result.get("tools", []) if result else []
        return self._tools

    async def call_tool(self, name: str, arguments: dict = None) -> str:
        """Call an MCP tool and return the text result."""
        if not self._initialized:
            return "MCP server not connected."

        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })
        if result is None:
            return "MCP tool call failed."

        # Extract text content from MCP response
        content = result.get("content", [])
        texts = []
        for block in content:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts) if texts else json.dumps(result)

    def has_tool(self, name: str) -> bool:
        """Check if a specific tool is available."""
        return any(t.get("name") == name for t in self._tools)

    @property
    def is_connected(self) -> bool:
        return self._initialized and self._process and self._process.returncode is None

    # ── Internal JSON-RPC transport ──────────────

    async def _send_request(self, method: str, params: dict, timeout: float = 30.0) -> Optional[dict]:
        """Send a JSON-RPC request and wait for the response."""
        self._request_id += 1
        rid = self._request_id
        message = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
            "params": params,
        }

        future = asyncio.get_event_loop().create_future()
        self._pending[rid] = future

        try:
            line = json.dumps(message) + "\n"
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            logger.error(f"MCP request timed out: {method}")
            return None
        except Exception as e:
            self._pending.pop(rid, None)
            logger.error(f"MCP request error ({method}): {e}")
            return None

    async def _send_notification(self, method: str, params: dict):
        """Send a JSON-RPC notification (no response expected)."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        try:
            line = json.dumps(message) + "\n"
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
        except Exception as e:
            logger.error(f"MCP notification error ({method}): {e}")

    async def _read_loop(self):
        """Read JSON-RPC responses from the MCP server's stdout."""
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    continue

                # Match response to pending request
                rid = msg.get("id")
                if rid is not None and rid in self._pending:
                    future = self._pending.pop(rid)
                    if "error" in msg:
                        logger.error(f"MCP error ({rid}): {msg['error']}")
                        if not future.done():
                            future.set_result(None)
                    else:
                        if not future.done():
                            future.set_result(msg.get("result", {}))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"MCP read loop error: {e}")


class CalendarMCPPlugin:
    """
    Calendar plugin backed by a Google Calendar MCP server.
    Drop-in replacement for CalendarPlugin — same async interface.

    Requires: npx and @anthropic/google-calendar-mcp (or compatible) installed.
    Configure via env vars:
        MCP_CALENDAR_COMMAND  — server command (default: npx)
        MCP_CALENDAR_ARGS     — JSON array of args (default: ["-y", "mcp-google-calendar"])
        MCP_CALENDAR_ENV      — JSON object of extra env vars to pass to the server
    """

    def __init__(self):
        self.command = os.getenv("MCP_CALENDAR_COMMAND", "npx")
        try:
            self.args = json.loads(os.getenv("MCP_CALENDAR_ARGS", '["-y", "mcp-google-calendar"]'))
        except json.JSONDecodeError:
            self.args = ["-y", "mcp-google-calendar"]
        try:
            self.extra_env = json.loads(os.getenv("MCP_CALENDAR_ENV", "{}"))
        except json.JSONDecodeError:
            self.extra_env = {}
        self.client = MCPClient(self.command, self.args, self.extra_env)
        self._started = False

    async def _ensure_connected(self) -> bool:
        """Start the MCP server if not already running."""
        if self.client.is_connected:
            return True
        if not self._started:
            self._started = True
            ok = await self.client.start()
            if ok:
                await self.client.list_tools()
                logger.info(f"Calendar MCP tools: {[t['name'] for t in self.client._tools]}")
            return ok
        return False

    async def create_event(self, params: dict) -> str:
        """Create a calendar event via MCP."""
        if not await self._ensure_connected():
            return "Calendar MCP server not available."

        # Map our params to standard MCP tool params
        tool_name = "create_event" if self.client.has_tool("create_event") else "createEvent"
        return await self.client.call_tool(tool_name, {
            "summary": params.get("title", "New Event"),
            "start": params.get("start_time", ""),
            "end": params.get("end_time", ""),
            "location": params.get("location", ""),
            "description": params.get("description", "Created by Mel AI Agent"),
            "timezone": params.get("timezone", "America/Chicago"),
        })

    async def get_events(self, params: dict) -> str:
        """List upcoming events via MCP."""
        if not await self._ensure_connected():
            return "Calendar MCP server not available."

        tool_name = "list_events" if self.client.has_tool("list_events") else "listEvents"
        return await self.client.call_tool(tool_name, {
            "days": params.get("days", 7),
            "maxResults": params.get("count", 10),
        })

    async def check_availability(self, params: dict) -> str:
        """Check if a time slot is free via MCP."""
        if not await self._ensure_connected():
            return "Calendar MCP server not available."

        tool_name = "check_availability" if self.client.has_tool("check_availability") else "freebusy"
        return await self.client.call_tool(tool_name, {
            "start": params.get("start_time", ""),
            "end": params.get("end_time", ""),
        })

    async def get_events_for_date(self, params: dict) -> str:
        """Get events for a specific date via MCP."""
        if not await self._ensure_connected():
            return "Calendar MCP server not available."

        tool_name = "list_events" if self.client.has_tool("list_events") else "listEvents"
        return await self.client.call_tool(tool_name, {
            "date": params.get("date", ""),
        })

    async def find_events_by_query(self, params: dict) -> list:
        """Search events by keyword via MCP. Returns text (not raw dicts)."""
        if not await self._ensure_connected():
            return []

        tool_name = "search_events" if self.client.has_tool("search_events") else "list_events"
        result = await self.client.call_tool(tool_name, {
            "query": params.get("query", ""),
            "date": params.get("date", ""),
        })
        # Return as list for compatibility with direct API (orchestrator expects list)
        return result if isinstance(result, list) else []

    async def delete_event(self, params: dict) -> str:
        """Delete a calendar event via MCP."""
        if not await self._ensure_connected():
            return "Calendar MCP server not available."

        tool_name = "delete_event" if self.client.has_tool("delete_event") else "deleteEvent"
        return await self.client.call_tool(tool_name, {
            "eventId": params.get("event_id", ""),
        })

    async def update_event(self, params: dict) -> str:
        """Update a calendar event via MCP."""
        if not await self._ensure_connected():
            return "Calendar MCP server not available."

        tool_name = "update_event" if self.client.has_tool("update_event") else "updateEvent"
        mcp_params = {"eventId": params.get("event_id", "")}
        if "title" in params:
            mcp_params["summary"] = params["title"]
        if "start_time" in params:
            mcp_params["start"] = params["start_time"]
        if "end_time" in params:
            mcp_params["end"] = params["end_time"]
        if "location" in params:
            mcp_params["location"] = params["location"]
        return await self.client.call_tool(tool_name, mcp_params)
