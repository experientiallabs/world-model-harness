//! Tool calls whose arguments fail to parse at their block stop, on wires
//! that reveal the stop reason only afterwards (Anthropic `message_delta`,
//! Bedrock `messageStop`). A provider-declared budget truncation makes the open
//! fragment the provider's own honest cut (dropped, stream Incomplete); any
//! other ending surfaces the parse failure as the malformed stream it is.

use super::{complete_streamed_tool, Normalizer};
use crate::errors::Failure;
use crate::events::{Event, ToolAccumulator};

impl Normalizer {
    /// Complete one tool call at its block stop when the stop reason is not
    /// yet known. A parse failure is HELD rather than raised: the provider may
    /// be about to say `max_tokens`, which makes the open fragment its own
    /// truncation (dropped, stream Incomplete), not a malformed stream.
    pub(super) fn complete_tool_deferring_failure(
        &mut self,
        index: u32,
        tool: &mut ToolAccumulator,
        events: &mut Vec<Event>,
    ) {
        let mut tool_events = Vec::new();
        match complete_streamed_tool(index, tool, &mut tool_events) {
            Ok(()) => events.extend(tool_events),
            Err(failure) => {
                if self.deferred_tool_failure.is_none() {
                    self.deferred_tool_failure = Some(failure);
                }
            }
        }
    }

    /// Resolve a held tool-argument failure at the terminal: a provider-declared
    /// budget truncation forgives it (the unfinished call was dropped), anything
    /// else surfaces it now.
    pub(super) fn resolve_deferred_tool_failure(&mut self, truncated: bool) -> Result<(), Failure> {
        match self.deferred_tool_failure.take() {
            Some(failure) if !truncated => Err(failure),
            _ => Ok(()),
        }
    }
}
