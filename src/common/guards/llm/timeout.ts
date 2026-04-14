import { type TimeoutConfig } from "../../types/index";

export class Timeout {
  private readonly timeoutMs: number;

  constructor(config: TimeoutConfig) {
    this.timeoutMs = config.timeoutMs;
  }

  public async execute<T>(
    func: (signal: AbortSignal) => Promise<T>
  ): Promise<T> {
    
    const controller = new AbortController();
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const timeoutPromise = new Promise<never>((_, reject) => {
      timeoutId = setTimeout(() => {
        // Construct and reject timeout error message
        const timeoutError = new Error(`Request timed out after ${this.timeoutMs}ms`);
        reject(timeoutError);

        // Trigger abort
        try {
          controller.abort(timeoutError);
        } catch {
          controller.abort();
        }
      }, this.timeoutMs);
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
