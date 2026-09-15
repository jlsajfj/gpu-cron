// The shipped model IS dist/index.js, and `make web` re-exports weights from $(CKPT).
// A dirty tree at publish time means the bundle may hold a checkpoint nobody reviewed:
// 0.1.0 shipped runs/pico instead of runs/pico7 exactly this way.
import { execFileSync } from 'node:child_process';

const dirty = execFileSync('git', ['status', '--porcelain'], { encoding: 'utf8' }).trim();
if (dirty) {
  console.error('refusing to publish: working tree is dirty\n');
  console.error(dirty);
  console.error('\ncommit or `git checkout` these first, then re-run `make web`.');
  process.exit(1);
}

const head = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
console.error(`publishing clean tree at ${head}`);
