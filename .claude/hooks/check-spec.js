#!/usr/bin/env node

/**
 * Spec-Loop PreToolUse Hook
 *
 * Configurable enforcement of spec-before-code rule.
 *
 * Modes (set in .spec-loop-config or CLAUDE.md):
 *   - "block"  : Block code changes until PROJECT.md is finalized
 *   - "ask"    : Prompt user to type "skip spec" to bypass (default)
 *   - "warn"   : Show warning but allow code changes
 *   - "off"    : Disable enforcement entirely
 */

import { existsSync, readFileSync } from 'fs';
import { join } from 'path';

// Files that don't require a spec (meta files, configs, spec files themselves)
const EXEMPT_PATTERNS = [
  /^PROJECT\.md$/i,
  /^ROADMAP\.md$/i,
  /^STATE\.md$/i,
  /^PLAN\.md$/i,
  /^ISSUES\.md$/i,
  /^CLAUDE\.md$/i,
  /^README\.md$/i,
  /^\.claude\//,
  /^\.git\//,
  /^\.spec-loop/,
  /package\.json$/,
  /package-lock\.json$/,
  /\.lock$/,
  /\.log$/,
  /node_modules\//,
];

const CONFIG_FILE = '.spec-loop-config';
const SKIP_FILE = '.spec-loop-skip';
const DEFAULT_MODE = 'ask';

function isExempt(filePath) {
  return EXEMPT_PATTERNS.some(pattern => pattern.test(filePath));
}

function isSkipEnabled(cwd) {
  const skipPath = join(cwd, SKIP_FILE);
  return existsSync(skipPath);
}

function getEnforcementMode(cwd) {
  const configPath = join(cwd, CONFIG_FILE);

  // Check .spec-loop-config file first
  if (existsSync(configPath)) {
    try {
      const config = JSON.parse(readFileSync(configPath, 'utf-8'));
      if (config.enforcement && ['block', 'ask', 'warn', 'off'].includes(config.enforcement)) {
        return config.enforcement;
      }
    } catch (e) {
      // Invalid config, continue checking
    }
  }

  // Check CLAUDE.md for spec-loop-enforcement directive
  const claudePath = join(cwd, 'CLAUDE.md');
  if (existsSync(claudePath)) {
    const content = readFileSync(claudePath, 'utf-8');
    const match = content.match(/spec-loop-enforcement:\s*(block|ask|warn|off)/i);
    if (match) {
      return match[1].toLowerCase();
    }
  }

  return DEFAULT_MODE;
}

function checkSpec() {
  const cwd = process.cwd();
  const projectPath = join(cwd, 'PROJECT.md');
  const mode = getEnforcementMode(cwd);

  // Read hook input from stdin
  let input = '';
  try {
    input = readFileSync(0, 'utf-8');
  } catch (e) {
    // No stdin, continue
  }

  let toolInput = {};
  try {
    toolInput = JSON.parse(input);
  } catch (e) {
    // Can't parse, allow through
    process.exit(0);
  }

  const filePath = toolInput.tool_input?.file_path || '';

  // Check if file is exempt
  if (isExempt(filePath)) {
    process.exit(0);
  }

  // If enforcement is off, allow everything
  if (mode === 'off') {
    process.exit(0);
  }

  // Check if PROJECT.md exists and is finalized
  const projectExists = existsSync(projectPath);
  let isFinalized = false;

  if (projectExists) {
    const content = readFileSync(projectPath, 'utf-8');
    isFinalized = content.includes('STATUS: FINALIZED');
  }

  // If spec is ready, allow through
  if (projectExists && isFinalized) {
    process.exit(0);
  }

  // Determine message based on what's missing
  let issue = '';
  let suggestion = '';

  if (!projectExists) {
    issue = 'No PROJECT.md found';
    suggestion = 'Run /spec init to create PROJECT.md';
  } else if (!isFinalized) {
    issue = 'PROJECT.md is not finalized';
    suggestion = 'Run /spec finalize to finalize your spec';
  }

  // Handle based on mode
  if (mode === 'warn') {
    // Output warning to stderr but allow through (exit 0)
    console.error(`⚠️  WARNING: ${issue}

The Spec-Loop Method recommends creating a specification before writing code.
${suggestion}

Proceeding anyway (enforcement mode: warn)
`);
    process.exit(0);
  }

  if (mode === 'ask') {
    // Check if skip file exists
    if (isSkipEnabled(cwd)) {
      console.error(`Skipping spec check (.spec-loop-skip file found)`);
      process.exit(0);
    }

    // Block but tell user they can bypass by creating skip file
    console.error(`BLOCKED: ${issue}

The Spec-Loop Method requires a project specification before writing code.

Options:
  1. ${suggestion} (recommended)
  2. Create a .spec-loop-skip file to bypass this check

To change enforcement mode, add to CLAUDE.md:
  spec-loop-enforcement: warn   (show warning only)
  spec-loop-enforcement: off    (disable entirely)
  spec-loop-enforcement: block  (no bypass allowed)
`);
    process.exit(1);
  }

  if (mode === 'block') {
    // Hard block, no bypass option
    console.error(`BLOCKED: ${issue}

The Spec-Loop Method requires a project specification before writing code.

${suggestion}

To change enforcement mode, add to CLAUDE.md:
  spec-loop-enforcement: ask    (allow "skip spec" bypass)
  spec-loop-enforcement: warn   (show warning only)
  spec-loop-enforcement: off    (disable entirely)
`);
    process.exit(1);
  }

  // Fallback - allow
  process.exit(0);
}

checkSpec();
