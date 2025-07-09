from datetime import datetime, timedelta, timezone


class TimestampPlugin:
    """Plugin that adds timestamps to user messages and tool results."""

    def __init__(self, timezone_offset=0, timezone_name="UTC"):
        """Initialize timestamp plugin with timezone settings.

        Parameters
        ----------
        timezone_offset : int, optional
            Minutes offset from UTC (default: 0)
        timezone_name : str, optional
            Timezone name for display (default: "UTC")
        """
        self.timezone_name = timezone_name
        self.timezone = timezone(-timedelta(minutes=timezone_offset))

    def format_timestamp(self) -> str:
        """Format timestamp with timezone adjustment."""
        dt = datetime.now(self.timezone)
        return dt.strftime(f"%Y-%m-%d %H:%M:%S {self.timezone_name}")

    async def hook_modify_user_message(self, message):
        """Add timestamp to user messages."""
        timestamp = self.format_timestamp()
        return (
            f"{message}\n\n<system message>current time: {timestamp}</system message>"
        )

    async def hook_modify_tool_result(self, tool_result):
        """Add timestamp to tool results."""
        timestamp = self.format_timestamp()
        tool_result["output"] = (
            f"{tool_result['output']}\n\n<system message>current time: {timestamp}</system message>"
        )
        return tool_result
