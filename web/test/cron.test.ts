import { describe, expect, it } from 'vitest';
import { CronExpressionParser } from 'cron-parser';
import { nextFireTimes, parseCronExpression } from '../src/cron.js';
import { isWellFormed } from '../src/automaton.js';

const LOCAL_TZ = Intl.DateTimeFormat().resolvedOptions().timeZone;

// All expectations are built from local date parts, so the assertions hold in any zone.
const JAN_1_2026 = new Date(2026, 0, 1, 0, 0, 0, 0);

function times(cron: string, count = 5, from: Date = JAN_1_2026): string[] {
  return nextFireTimes(cron, count, from).map((d) => d.toISOString());
}

function localTime(
  y: number,
  mo: number,
  d: number,
  h: number,
  mi: number,
): string {
  return new Date(y, mo - 1, d, h, mi, 0, 0).toISOString();
}

describe('parseCronExpression', () => {
  it('reads every term shape in the dialect', () => {
    const parsed = parseCronExpression('*/15 0-23/2 1,15 1-6 0');
    expect(parsed).not.toBeNull();
    expect([...parsed!.minute]).toEqual([0, 15, 30, 45]);
    expect([...parsed!.hour]).toEqual([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]);
    expect([...parsed!.dom]).toEqual([1, 15]);
    expect([...parsed!.month]).toEqual([1, 2, 3, 4, 5, 6]);
    expect([...parsed!.dow]).toEqual([0]);
    expect(parsed!.domWildcard).toBe(false);
    expect(parsed!.dowWildcard).toBe(false);
  });

  it('flags bare stars and rejects anything else', () => {
    expect(parseCronExpression('* * * * *')!.domWildcard).toBe(true);
    expect(parseCronExpression('* * * * *')!.dowWildcard).toBe(true);
    expect(parseCronExpression('* * */2 * *')!.domWildcard).toBe(false);
    for (const bad of ['', '* * * *', '60 * * * *', '0 24 * * *', '0 0 0 * *', 'a * * * *']) {
      expect(parseCronExpression(bad), bad).toBeNull();
    }
  });
});

describe('nextFireTimes', () => {
  it('hand computed cases', () => {
    expect(times('0 0 * * *', 3)).toEqual([
      localTime(2026, 1, 2, 0, 0),
      localTime(2026, 1, 3, 0, 0),
      localTime(2026, 1, 4, 0, 0),
    ]);
    expect(times('*/15 * * * *', 5)).toEqual([
      localTime(2026, 1, 1, 0, 15),
      localTime(2026, 1, 1, 0, 30),
      localTime(2026, 1, 1, 0, 45),
      localTime(2026, 1, 1, 1, 0),
      localTime(2026, 1, 1, 1, 15),
    ]);
    // Jan 1 2026 is a Thursday, so a weekday schedule fires Thu, Fri then Mon.
    expect(times('0 9 * * 1-5', 5)).toEqual([
      localTime(2026, 1, 1, 9, 0),
      localTime(2026, 1, 2, 9, 0),
      localTime(2026, 1, 5, 9, 0),
      localTime(2026, 1, 6, 9, 0),
      localTime(2026, 1, 7, 9, 0),
    ]);
    expect(times('0 0 1 * *', 3)).toEqual([
      localTime(2026, 2, 1, 0, 0),
      localTime(2026, 3, 1, 0, 0),
      localTime(2026, 4, 1, 0, 0),
    ]);
    expect(times('0 0 */2 * *', 3)).toEqual([
      localTime(2026, 1, 3, 0, 0),
      localTime(2026, 1, 5, 0, 0),
      localTime(2026, 1, 7, 0, 0),
    ]);
    // Both day fields are restricted, so the day of month wins on the 1st and the day of
    // week on every Monday: the OR rule, not the AND.
    expect(times('0 0 1 * 1', 3)).toEqual([
      localTime(2026, 1, 5, 0, 0),
      localTime(2026, 1, 12, 0, 0),
      localTime(2026, 1, 19, 0, 0),
    ]);
    expect(times('0 12 1,15 * *', 2)).toEqual([
      localTime(2026, 1, 1, 12, 0),
      localTime(2026, 1, 15, 12, 0),
    ]);
    expect(times('0 0 29 2 *', 2)).toEqual([
      localTime(2028, 2, 29, 0, 0),
      localTime(2032, 2, 29, 0, 0),
    ]);
  });

  it('is strictly after the reference time and ignores sub minute parts', () => {
    expect(times('*/15 * * * *', 1, new Date(2026, 0, 1, 0, 15, 0, 0))).toEqual([
      localTime(2026, 1, 1, 0, 30),
    ]);
    expect(times('*/15 * * * *', 1, new Date(2026, 0, 1, 0, 15, 0, 1))).toEqual([
      localTime(2026, 1, 1, 0, 30),
    ]);
    expect(times('*/15 * * * *', 1, new Date(2026, 0, 1, 0, 14, 59, 999))).toEqual([
      localTime(2026, 1, 1, 0, 15),
    ]);
  });

  it('returns fewer times when the expression can never fire', () => {
    expect(nextFireTimes('0 0 30 2 *', 5, JAN_1_2026)).toEqual([]);
    expect(nextFireTimes('not cron', 5, JAN_1_2026)).toEqual([]);
  });

  it('agrees with cron-parser on every expression the automaton can emit', () => {
    const crons = [
      '0 9 * * 1-5',
      '*/15 * * * *',
      '0 0 * * *',
      '0 0 1 * *',
      '0 0 */2 * *',
      '0 0 1 * 1',
      '0 0 1,15 * *',
      '0 12 * * 0',
      '30 6 1,15 1-6 *',
      '5 4 * * 0',
      '0 22 * * 1-5',
      '0 0 1 1 *',
      '*/5 9-17 * * 1-5',
      '0 0,12 * * *',
      '0 0 * 2,4,6,8,10,12 *',
      '0 0 * * 0,1,2,3,4,5,6',
      '15,45 8-18/4 1-7 * *',
      '30 0 1 1,7 *',
    ];
    const from = new Date(2026, 5, 15, 13, 37, 0, 0);
    const mismatches: string[] = [];
    for (const cron of crons) {
      expect(isWellFormed(cron), cron).toBe(true);
      const mine = times(cron, 8, from);
      const iterator = CronExpressionParser.parse(cron, {
        currentDate: from,
        tz: LOCAL_TZ,
      });
      const theirs: string[] = [];
      for (let i = 0; i < 8; i += 1) theirs.push(iterator.next().toDate().toISOString());
      if (JSON.stringify(mine) !== JSON.stringify(theirs)) {
        mismatches.push(`${cron}: got ${mine.join(',')} want ${theirs.join(',')}`);
      }
    }
    expect(mismatches).toEqual([]);
  });
});
