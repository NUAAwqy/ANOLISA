import type { AgentMessage } from "./openclaw-bridge.js";
import { estimateTokens } from "./estimate-tokens.js";

export interface AssembleInput {
  messages: AgentMessage[];
  summaries: Map<number, string>;
  tokenBudget: number;
  freshTailTurns: number;
}

export interface AssembleResult {
  messages: AgentMessage[];
  estimatedTokens: number;
  stats: {
    totalTurns: number;
    freshTailTurnCount: number;
    freshTailMessageCount: number;
    summaryCount: number;
    droppedTurns: number;
  };
}

interface Turn {
  turnSeq: number;
  messages: AgentMessage[];
  tokenCount: number;
}

function deriveTurns(messages: AgentMessage[]): Turn[] {
  const turns: Turn[] = [];
  let turnSeq = 0;
  let current: Turn | null = null;

  for (const msg of messages) {
    const role = normalizeRole(msg.role);
    if (role === "user" || role === "system") {
      turnSeq++;
      current = { turnSeq, messages: [], tokenCount: 0 };
      turns.push(current);
    }
    if (!current) {
      turnSeq = 1;
      current = { turnSeq, messages: [], tokenCount: 0 };
      turns.push(current);
    }
    const tokens = estimateTokens(JSON.stringify(msg));
    current.messages.push(msg);
    current.tokenCount += tokens;
  }

  return turns;
}

function normalizeRole(role: string): string {
  if (role === "toolResult" || role === "tool_result") return "tool";
  return role;
}

function tailBudgetError(tailTokens: number, tokenBudget: number): Error {
  return new Error(
    `selective-claw: protected fresh tail (${tailTokens} tokens) exceeds the token budget ` +
      `(${tokenBudget}); the context cannot be assembled without truncating tool payloads`,
  );
}

export class Assembler {
  constructor(private defaultFreshTailTurns: number = 3) {}

  assemble(input: AssembleInput): AssembleResult {
    const freshTailTurns = input.freshTailTurns ?? this.defaultFreshTailTurns;
    const turns = deriveTurns(input.messages);

    if (turns.length === 0) {
      return {
        messages: [],
        estimatedTokens: 0,
        stats: { totalTurns: 0, freshTailTurnCount: 0, freshTailMessageCount: 0, summaryCount: 0, droppedTurns: 0 },
      };
    }

    if (turns.length <= freshTailTurns) {
      const allTokens = turns.reduce((s, t) => s + t.tokenCount, 0);
      if (allTokens > input.tokenBudget) {
        throw tailBudgetError(allTokens, input.tokenBudget);
      }
      return {
        messages: input.messages,
        estimatedTokens: allTokens,
        stats: {
          totalTurns: turns.length,
          freshTailTurnCount: turns.length,
          freshTailMessageCount: input.messages.length,
          summaryCount: 0,
          droppedTurns: 0,
        },
      };
    }

    const tail = turns.slice(-freshTailTurns);
    const older = turns.slice(0, -freshTailTurns);

    // The fresh tail is protected: complete recent messages are never cut to
    // manufacture a fitting context. When the tail alone exceeds the budget
    // the assembly is rejected explicitly instead of returned over budget.
    const tailMessages: AgentMessage[] = [];
    for (const turn of tail) {
      tailMessages.push(...turn.messages);
    }
    const tailTokens = tailMessages.reduce(
      (total, message) => total + estimateTokens(JSON.stringify(message)),
      0,
    );
    if (tailTokens > input.tokenBudget) {
      throw tailBudgetError(tailTokens, input.tokenBudget);
    }

    const summaryLines: string[] = [];
    let droppedTurns = 0;
    for (const turn of older) {
      const summary = input.summaries.get(turn.turnSeq);
      if (summary) {
        summaryLines.push(`Turn ${turn.turnSeq}: ${summary}`);
      } else {
        droppedTurns++;
        const preview = turn.messages
          .map((m) => extractText(m).slice(0, 60))
          .join(" | ");
        summaryLines.push(`Turn ${turn.turnSeq}: [no summary] ${preview}...`);
      }
    }

    // Admit whole summary entries newest-first while the joined summary
    // message still fits the remaining estimate; an entry that cannot fit is
    // skipped, never truncated.
    const header = "[summary] Earlier conversation context:";
    const admitted: string[] = [];
    for (const line of [...summaryLines].reverse()) {
      const candidate: AgentMessage = {
        role: "user",
        content: [header, ...admitted, line].join("\n"),
      };
      if (estimateTokens(JSON.stringify(candidate)) + tailTokens <= input.tokenBudget) {
        admitted.push(line);
      }
    }

    const result: AgentMessage[] = [];

    if (admitted.length > 0) {
      result.push({
        role: "user",
        content: [header, ...admitted.reverse()].join("\n"),
      });
    }

    result.push(...tailMessages);

    const estimatedTokens = result.reduce(
      (total, message) => total + estimateTokens(JSON.stringify(message)),
      0,
    );

    return {
      messages: result,
      estimatedTokens,
      stats: {
        totalTurns: turns.length,
        freshTailTurnCount: tail.length,
        freshTailMessageCount: tailMessages.length,
        summaryCount: admitted.length,
        droppedTurns,
      },
    };
  }
}

function extractText(msg: AgentMessage): string {
  if (typeof msg.content === "string") return msg.content;
  if (Array.isArray(msg.content)) {
    return msg.content
      .map((block: any) => {
        if (typeof block === "string") return block;
        if (block?.type === "text" && typeof block.text === "string") return block.text;
        return "";
      })
      .join(" ");
  }
  return "";
}
