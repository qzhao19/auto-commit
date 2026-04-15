import { type TimeoutConfig, type TimeoutExecuteOptions } from "../../types/index";

export class Timeout {
  private readonly defaultTimeoutMs: number;

  constructor(config: TimeoutConfig) {
    this.defaultTimeoutMs = config.timeoutMs;
  }

  public async execute<T>(
    func: (signal: AbortSignal) => Promise<T>,
    options?: TimeoutExecuteOptions
  ): Promise<T> {

    const timeoutMs = options?.timeoutMs ?? this.defaultTimeoutMs;

    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new Error("[Timeout] Request timed out before execution started");
    }
    
    const controller = new AbortController();
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const timeoutPromise = new Promise<never>((_, reject) => {
      timeoutId = setTimeout(() => {
        // Construct and reject timeout error message
        const timeoutError = new Error(`[Timeout] Request timed out after ${timeoutMs}ms`);
        reject(timeoutError);

        // Trigger abort
        try {
          controller.abort(timeoutError);
        } catch {
          controller.abort();
        }
      }, timeoutMs);
    });

    const taskPromise = Promise.resolve().then(() => func(controller.signal));

    try {
      return await Promise.race([taskPromise, timeoutPromise]);
    } finally {
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
      }
    }
  }
}
