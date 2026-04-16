import { type LLModelParams } from "./model-settings";
import { type RequestSafetyConfig } from "./request-safety";

export interface RuntimeConfig {
  llm: {
    provider: string;
    model: string;
    modelParams: LLModelParams;
  };
  requestSafty: {
    retry: NonNullable<RequestSafetyConfig["retry"]>;
    timeout: NonNullable<RequestSafetyConfig["timeout"]>;
    rateLimiter: NonNullable<RequestSafetyConfig["rateLimiter"]>;
  };
}
