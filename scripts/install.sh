#!/bin/sh
# Install verified official release bytes into the current user's directory.
set -eu
LC_ALL=C
export LC_ALL

die() { printf 'Ferrum installer: %s\n' "$*" >&2; exit 1; }
note() { printf '%s\n' "$*" >&2; }
checksum() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    else shasum -a 256 "$1" | awk '{print $1}'; fi
}
read_checksum() {
    awk -v name="$2" 'NR == 1 && NF == 2 && $2 == name && length($1) == 64 && $1 !~ /[^0-9a-f]/ {hash=$1} END {if(NR != 1 || hash == "") exit 1; print hash}' "$1"
}
download() {
    curl --fail --silent --show-error --location --retry 2 --connect-timeout 20 \
        --max-time 600 --proto "$protocols" --proto-redir '=https' "$1" -o "$2" \
        || die "Download failed: $1"
}
cleanup() {
    if [ -n "$work" ]; then rm -rf "$work"; fi
    if [ "$locked" = yes ]; then rm -f "$lock/pid"; rmdir "$lock"; fi
}
prepare() {
    asset="ferrum-$platform"
    [ "$backend" != cuda ] || asset="$asset-cuda-sm89"
    asset="$asset.tar.gz"
    note "Downloading Ferrum $version ($backend, $platform)..."
    download "$release_base/v$version/$asset" "$work/$asset"
    download "$release_base/v$version/$asset.sha256" "$work/asset.sha256"
    download "$release_base/v$version/$asset.binary.sha256" "$work/binary.sha256"
    asset_hash=$(read_checksum "$work/asset.sha256" "$asset") || die 'Invalid archive checksum file'
    [ "$(checksum "$work/$asset")" = "$asset_hash" ] || die 'Archive SHA-256 mismatch'
    binary_hash=$(read_checksum "$work/binary.sha256" ferrum) || die 'Invalid binary checksum file'
    tar -tzf "$work/$asset" > "$work/members" || die 'Cannot list release archive'
    printf '%s\n' LICENSE README.md ferrum > "$work/expected"
    [ "$backend" != cuda ] || printf '%s\n' CUDA-BUILD.txt >> "$work/expected"
    sort "$work/members" > "$work/members.sorted"
    sort "$work/expected" > "$work/expected.sorted"
    cmp -s "$work/members.sorted" "$work/expected.sorted" || die 'Unexpected or duplicate archive member'
    tar -tvzf "$work/$asset" > "$work/member-types" || die 'Cannot inspect archive member types'
    awk 'substr($0,1,1) != "-" {exit 1}' "$work/member-types" || die 'Archive members must be regular files'
    unpack="$work/payload-$backend"
    mkdir "$unpack"
    # Write only the approved flat names, without restoring owners or permissions.
    while IFS= read -r member; do
        tar -xOzf "$work/$asset" "$member" > "$unpack/$member" || die "Cannot extract $member"
    done < "$work/expected"
    [ "$(checksum "$unpack/ferrum")" = "$binary_hash" ] || die 'Binary SHA-256 mismatch'
    chmod 755 "$unpack/ferrum"
    runtime_ok=no
    if actual_version=$("$unpack/ferrum" --version 2> "$work/runtime-error"); then
        [ "$actual_version" = "ferrum $version" ] || die 'Downloaded binary version differs from requested release'
        runtime_ok=yes
    else
        cat "$work/runtime-error" >&2
    fi
}
profile_check() {
    [ ! -L "$1" ] || die "Refusing to edit a symlinked shell profile: $1"
    [ ! -e "$1" ] || [ -f "$1" ] || die "Shell profile is not a regular file: $1"
    if [ -f "$1" ] && grep -Fq '# >>> Ferrum installer PATH >>>' "$1"; then
        [ "$(grep -Fxc '# >>> Ferrum installer PATH >>>' "$1")" = 1 ] && \
        [ "$(grep -Fxc '. "$HOME/.local/share/ferrum/installer/env"' "$1")" = 1 ] && \
        [ "$(grep -Fxc '# <<< Ferrum installer PATH <<<' "$1")" = 1 ] \
            || die "Ferrum's PATH fragment was edited; preserve it and configure PATH manually: $1"
    fi
}
profile_add() {
    if [ -f "$1" ] && grep -Fq '# >>> Ferrum installer PATH >>>' "$1"; then return; fi
    printf '\n%s\n%s\n%s\n' '# >>> Ferrum installer PATH >>>' \
        '. "$HOME/.local/share/ferrum/installer/env"' '# <<< Ferrum installer PATH <<<' >> "$1"
}
main() {
    backend=auto; requested=auto; version=latest; modify_path=yes
    release_base=https://github.com/sizzlecar/ferrum-infer-rs/releases/download
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --backend|--version|--release-base-url)
                [ "$#" -ge 2 ] || die "Missing value for $1"
                case "$1" in --backend) backend=$2;; --version) version=${2#v};; --release-base-url) release_base=${2%/};; esac
                shift 2;;
            --no-modify-path) modify_path=no; shift;;
            --help|-h)
                printf '%s\n' 'Usage: sh install.sh [--backend auto|cpu|metal|cuda] [--version VERSION]' \
                    '  --no-modify-path       Leave shell profiles untouched' \
                    '  --release-base-url URL  Explicit release mirror (vVERSION/asset layout)' \
                    'Installs to ~/.local/bin; preserves unmanaged Ferrum installations.' \
                    'Linux CUDA requires sm89 GPU, NVIDIA driver, CUDA 12.4 and NCCL runtimes.'
                return;;
            *) die "Unknown option: $1";;
        esac
    done
    requested=$backend
    case "$backend" in auto|cpu|metal|cuda) ;; *) die 'Backend must be auto, cpu, metal, or cuda';; esac
    case "${HOME:-}" in /*) ;; *) die 'HOME must be an absolute user directory';; esac
    case "$HOME" in *:*|*'
'*) die 'HOME cannot contain a PATH separator or newline';; esac
    for tool in curl tar awk sort cmp grep sed uname mktemp readlink; do command -v "$tool" >/dev/null 2>&1 || die "Required command is missing: $tool"; done
    command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 || die 'sha256sum or shasum is required'
    protocols='=https'
    case "$release_base" in
        https://*) ;;
        http://127.0.0.1:*|http://localhost:*)
            printf '%s\n' "$release_base" | grep -Eq '^http://(127\.0\.0\.1|localhost):[0-9]+(/[A-Za-z0-9._/-]*)?$' || die 'Invalid loopback mirror URL'
            protocols='=http,https';;
        *) die 'Release mirror must use HTTPS (HTTP is allowed only for explicit loopback testing)';;
    esac
    if [ "$version" = latest ]; then
        [ "$release_base" = https://github.com/sizzlecar/ferrum-infer-rs/releases/download ] || die 'An explicit mirror requires --version'
        latest=$(curl --fail --silent --show-error --location --head --connect-timeout 20 --max-time 60 \
            --proto '=https' --proto-redir '=https' --output /dev/null --write-out '%{url_effective}' \
            https://github.com/sizzlecar/ferrum-infer-rs/releases/latest) || die 'Cannot resolve latest release'
        case "$latest" in https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/v*) version=${latest##*/v};; *) die 'Unexpected latest release redirect';; esac
    fi
    printf '%s\n' "$version" | grep -Eq '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$' || die 'Version must be a formal MAJOR.MINOR.PATCH release'
    os=$(uname -s); arch=$(uname -m)
    case "$os/$arch" in
        Darwin/arm64|Darwin/aarch64)
            platform=macos-aarch64
            case "$backend" in auto|metal) backend=metal;; *) die 'The macOS release is Apple Silicon with Metal';; esac;;
        Linux/x86_64|Linux/amd64)
            platform=linux-x86_64
            gpu_supported=no
            if command -v nvidia-smi >/dev/null 2>&1; then
                if caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null); then
                    if printf '%s\n' "$caps" | awk 'BEGIN{n=0} {gsub(/[[:space:]]/,""); if($0!="8.9") exit 1; n++} END{if(n==0) exit 1}'; then gpu_supported=yes; fi
                fi
            fi
            case "$backend" in
                auto) if [ "$gpu_supported" = yes ]; then backend=cuda; else backend=cpu; fi;;
                cuda) [ "$gpu_supported" = yes ] || die 'CUDA sm89 release requires an actual compute-capability 8.9 GPU';;
                cpu) ;; *) die 'Metal is available only on Apple Silicon macOS';;
            esac;;
        *) die "No official binary for $os/$arch";;
    esac
    root="$HOME/.local/share/ferrum/installer"; bindir="$HOME/.local/bin"; binary="$bindir/ferrum"
    active=$(command -v ferrum 2>/dev/null || :)
    [ -z "$active" ] || [ "$active" = "$binary" ] || die "Existing Ferrum is managed elsewhere ($active); it was not changed"
    [ ! -L "$bindir" ] || die 'Refusing a symlinked ~/.local/bin directory'
    previous_link=
    if [ -e "$binary" ] || [ -L "$binary" ]; then
        [ -L "$binary" ] || die "Unmanaged binary already exists: $binary"
        current=$(readlink "$binary")
        case "$current" in "$root"/releases/*/ferrum) ;; *) die "Unmanaged Ferrum link already exists: $binary";; esac
        current_dir=${current%/ferrum}; current_name=${current_dir##*/}
        [ "$current_dir" = "$root/releases/$current_name" ] && [ "$current_name" != .. ] && [ "$current_name" != . ] || die 'Invalid managed installation path'
        [ ! -L "$current_dir" ] && [ ! -L "$current" ] && [ -f "$current" ] || die 'Managed installation was replaced'
        old_hash=$(read_checksum "$current_dir/.binary.sha256" ferrum) || die 'Missing ownership checksum; existing binary preserved'
        [ "$(checksum "$current")" = "$old_hash" ] || die 'Existing managed binary was edited; it was preserved'
        previous_link=$current
    fi
    profile_a=; profile_b=
    if [ "$modify_path" = yes ]; then
        shell_name=${SHELL:-}
        case "${shell_name##*/}" in
            zsh) profile_a="$HOME/.zshrc";;
            bash)
                profile_a="$HOME/.bashrc"
                if [ -e "$HOME/.bash_profile" ]; then profile_b="$HOME/.bash_profile"
                elif [ -e "$HOME/.bash_login" ]; then profile_b="$HOME/.bash_login"
                else profile_b="$HOME/.profile"; fi;;
            sh|dash|'') profile_a="$HOME/.profile";;
            *) die 'This shell requires manual PATH setup; pass --no-modify-path';;
        esac
        profile_check "$profile_a"
        [ -z "$profile_b" ] || profile_check "$profile_b"
    fi
    if [ -e "$root" ] || [ -L "$root" ]; then
        [ ! -L "$root" ] && [ -d "$root" ] && [ "$(cat "$root/.owner" 2>/dev/null)" = ferrum-unix-installer-v1 ] || die 'Existing installer directory is not owned by this installer'
    else
        mkdir -p "$HOME/.local/share/ferrum"
        mkdir "$root"
        printf '%s\n' ferrum-unix-installer-v1 > "$root/.owner"
    fi
    lock="$root/.lock"; locked=no; work=
    mkdir "$lock" 2>/dev/null || die "Another or interrupted installer owns $lock; inspect its pid before retrying"
    locked=yes; printf '%s\n' "$$" > "$lock/pid"
    trap cleanup EXIT
    trap 'exit 1' HUP INT TERM
    work=$(mktemp -d "$root/download.XXXXXXXX")
    prepare
    if [ "$runtime_ok" != yes ]; then
        if [ "$requested" = auto ] && [ "$backend" = cuda ]; then
            note 'CUDA binary could not start with installed runtimes; selecting CPU. CUDA 12.4/NCCL/driver are not bundled.'
            backend=cpu; prepare
        fi
        [ "$runtime_ok" = yes ] || die 'The release binary cannot start on this host; no installed binary was replaced'
    fi
    [ ! -L "$root/releases" ] || die 'Managed releases directory was replaced by a symlink'
    mkdir -p "$root/releases" "$bindir"
    destination="$root/releases/$version-$backend-$binary_hash"
    if [ -e "$destination" ] || [ -L "$destination" ]; then
        [ ! -L "$destination" ] && [ -d "$destination" ] || die 'Existing version path was replaced'
        while IFS= read -r member; do
            [ ! -L "$destination/$member" ] && cmp -s "$unpack/$member" "$destination/$member" || die "Existing version file was edited: $member"
        done < "$work/expected"
        [ "$(read_checksum "$destination/.binary.sha256" ferrum)" = "$binary_hash" ] || die 'Existing version ownership checksum differs'
    else
        cp "$work/binary.sha256" "$unpack/.binary.sha256"
        cp "$work/asset.sha256" "$unpack/.asset.sha256"
        mv "$unpack" "$destination"
    fi
    cat > "$work/env" <<'ENV'
# Managed by the Ferrum user installer.
case ":${PATH-}:" in
    *":$HOME/.local/bin:"*) ;;
    *) export PATH="$HOME/.local/bin${PATH:+:$PATH}" ;;
esac
ENV
    [ ! -L "$root/env" ] || die 'Managed environment file was replaced by a symlink'
    if [ -e "$root/env" ]; then cmp -s "$work/env" "$root/env" || die 'Managed environment file was edited; it was preserved'; fi
    if [ -n "$previous_link" ]; then
        [ -L "$binary" ] && [ "$(readlink "$binary")" = "$previous_link" ] && [ "$(checksum "$previous_link")" = "$old_hash" ] || die 'Existing installation changed during download; no replacement attempted'
    else
        [ ! -e "$binary" ] && [ ! -L "$binary" ] || die 'Another installation appeared during download; it was preserved'
    fi
    if [ "$modify_path" = yes ]; then profile_check "$profile_a"; [ -z "$profile_b" ] || profile_check "$profile_b"; fi
    mv -f "$work/env" "$root/env"
    # Finish fallible profile writes before committing the new launch target.
    if [ "$modify_path" = yes ]; then profile_add "$profile_a"; [ -z "$profile_b" ] || profile_add "$profile_b"; fi
    ln -s "$destination/ferrum" "$work/ferrum-link"
    mv -f "$work/ferrum-link" "$binary"
    note "Installed ferrum $version ($backend) at $binary"
    note 'Open a new terminal, or run: . "$HOME/.local/share/ferrum/installer/env"'
}
main "$@"
