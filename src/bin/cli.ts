import { AutoCommit } from "../cli/index";
import {
  GitCode,
  GitError,
  ProviderError,
  ConfigError,
} from "../shared/exceptions/index";

async function main(): Promise<void> {
  await new AutoCommit().run();
}

main().catch((error: unknown) => {
  if (error instanceof GitError) {
    if (
      error.code === GitCode.STAGING_EMPTY ||
      error.code === GitCode.NOTHING_TO_COMMIT
    ) {
      console.log(error.message);
      process.exit(0);
    }
    console.error(`${error.message}`);
    process.exit(1);
  }

  if (error instanceof ProviderError) {
    console.error(`LLM error [${error.code}]: ${error.message}`);
    process.exit(1);
  }

  if (error instanceof ConfigError) {
    console.error(`Config error [${error.code}]: ${error.message}`);
    process.exit(1);
  }

  // Non-TTY: readKey() throws a plain Error with this message
  if (error instanceof Error && error.message === "Run in an interactive terminal") {
    console.error(`${error.message}`);
    process.exit(1);
  }

  // Unknown errors — show full stack if AUTOCOMMIT_VERBOSE=true
  const verbose = process.env["AUTOCOMMIT_VERBOSE"] === "true";
  if (verbose && error instanceof Error && error.stack) {
    console.error(error.stack);
  } else {
    console.error(
      `Unexpected error: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  process.exit(1);
});
