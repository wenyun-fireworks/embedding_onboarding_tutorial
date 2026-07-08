# Sourced by the step scripts. Loads KEY=VALUE lines from .env WITHOUT
# overwriting variables already present in the environment, so you can either
# put values in .env or export them (exports win).
_load_env() {
  local envfile="$1"
  [ -f "$envfile" ] || return 0
  local line key val
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue;; esac
    [ "${line#*=}" = "$line" ] && continue   # no '=' -> skip
    key="${line%%=*}"
    val="${line#*=}"
    key="$(printf '%s' "$key" | tr -d '[:space:]')"
    [ -z "$key" ] && continue
    # strip surrounding quotes
    val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
    if [ -z "$(eval "printf '%s' \"\${$key:-}\"")" ]; then
      export "$key=$val"
    fi
  done < "$envfile"
}
