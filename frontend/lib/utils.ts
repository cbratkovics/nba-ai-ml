/** Join class names, skipping falsy entries (the only helper the components need). */
export function cn(...inputs: Array<string | false | null | undefined>): string {
  return inputs.filter(Boolean).join(' ')
}
