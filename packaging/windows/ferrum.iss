; Wrap the directory produced by `release_delivery portable inspect --extract-dir`.
; The caller must first accept that payload's manifest, hashes and GPU checks.
; ISCC.exe "/DPayloadDir=C:\accepted payload" "/DLauncherPath=C:\tools\ferrum-launcher.exe" /DAppVersion=0.8.9 /O"C:\artifacts" ferrum.iss
; Use Inno Setup 6.3 or later; record the actual ISCC version with the artifact.
; https://jrsoftware.org/ishelp/topic_isppcc.htm
; https://jrsoftware.org/ishelp/topic_compilercmdline.htm

#ifndef PayloadDir
  #error PayloadDir must name the accepted, extracted portable payload
#endif
#ifndef AppVersion
  #error AppVersion must be the numeric release version, without a v prefix
#endif
#ifndef LauncherPath
  #error LauncherPath must name the stable native Ferrum launcher
#endif
#ifndef Backend
  #define Backend "cuda-sm89"
#endif
#if Backend == "cpu"
  #define BackendLabel "CPU"
  #define BackendDescription "CPU inference without NVIDIA driver or CUDA runtime requirements."
#elif Backend == "cuda-sm89"
  #define BackendLabel "CUDA sm89"
  #define BackendDescription "Inference for NVIDIA CUDA compute capability 8.9 (sm89)."
#else
  #error Backend must be cpu or cuda-sm89
#endif
#if !FileExists(LauncherPath)
  #error LauncherPath does not exist
#endif
#if !FileExists(PayloadDir + "\ferrum.exe")
  #error PayloadDir does not contain ferrum.exe
#endif
#if !FileExists(PayloadDir + "\ferrum-portable.json")
  #error PayloadDir does not contain ferrum-portable.json
#endif
#if FileExists(PayloadDir + "\nvcuda.dll")
  #error The NVIDIA system driver must not be included in the payload
#endif
#define PayloadManifestHash GetSHA256OfFile(PayloadDir + "\ferrum-portable.json")
#define VersionDir AppVersion + "-" + PayloadManifestHash
#define LauncherHash GetSHA256OfFile(LauncherPath)

[Setup]
; Keep this ID stable across upgrades. This is one installation per user.
AppId=Ferrum.CLI.Windows
AppName=Ferrum ({#BackendLabel})
AppVersion={#AppVersion}
AppPublisher=Ferrum
AppPublisherURL=https://github.com/sizzlecar/ferrum-infer-rs
AppComments={#BackendDescription}
DefaultDirName={localappdata}\Programs\Ferrum
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0
UninstallDisplayIcon={app}\ferrum.exe
UninstallDisplayName=Ferrum ({#BackendLabel})
ChangesEnvironment=yes
SetupMutex=Ferrum.CLI.Windows.Setup
; Never stop or restart a user's inference process automatically.
CloseApplications=no
RestartApplications=no
OutputDir={#PayloadDir}\..\installer
OutputBaseFilename=ferrum-{#AppVersion}-windows-x86_64-{#Backend}-setup
VersionInfoVersion={#AppVersion}
VersionInfoProductName=Ferrum ({#BackendLabel})
VersionInfoDescription=Ferrum current-user installer (Windows x86_64, {#BackendLabel})
VersionInfoProductTextVersion={#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Files]
; The stable entry point is installed once. Engine upgrades never replace it.
Source: "{#LauncherPath}"; DestDir: "{app}"; DestName: "ferrum.exe"; Flags: onlyifdoesntexist
; Each complete payload is immutable. Existing matching files may be running.
; PrepareToInstall verifies every existing file before onlyifdoesntexist skips it.
Source: "{#PayloadDir}\ferrum.exe"; DestDir: "{app}\versions\{#VersionDir}"; Flags: onlyifdoesntexist
Source: "{#PayloadDir}\*.dll"; DestDir: "{app}\versions\{#VersionDir}"; Flags: onlyifdoesntexist
Source: "{#PayloadDir}\ferrum-portable.json"; DestDir: "{app}\versions\{#VersionDir}"; Flags: onlyifdoesntexist
Source: "{#PayloadDir}\licenses\*"; DestDir: "{app}\versions\{#VersionDir}\licenses"; Flags: onlyifdoesntexist

[Messages]
FinishedLabel=Ferrum ({#BackendLabel}) is installed.%n%nOpen a new terminal to use ferrum run or ferrum serve. Models are selected separately with the command-line options.

[Code]
const
  EnvironmentKey = 'Environment';
  OwnershipKey = 'Software\Ferrum\Installer';
  VersionOwnershipKey = 'Software\Ferrum\Installer\Versions';
  OwnershipValue = 'UserPathSuffix';
  PathExistedValue = 'UserPathOriginallyExisted';
  StringTypesWithoutExpansion = $10000006; // RRF_NOEXPAND | RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ
  RegistryExpandString = 2; // REG_EXPAND_SZ

var
  UserPathConfigured: Boolean;

function MoveFileEx(ExistingFile, NewFile: String; Flags: Cardinal): Boolean;
  external 'MoveFileExW@kernel32.dll stdcall';

function GetFileAttributes(Path: String): Cardinal;
  external 'GetFileAttributesW@kernel32.dll stdcall';

function GetTempFileName(Directory, Prefix: String; Unique: Cardinal; FileName: String): Cardinal;
  external 'GetTempFileNameW@kernel32.dll stdcall';

procedure RequireOrdinaryDirectory(const Path: String);
var
  Attributes: Cardinal;
begin
  Attributes := GetFileAttributes(Path);
  if Attributes = $FFFFFFFF then
    Exit;
  if ((Attributes and $10) = 0) or ((Attributes and $400) <> 0) then
    RaiseException('Ferrum installation directory is not an ordinary directory: ' + Path);
end;

function VersionPath: String;
begin
  Result := ExpandConstant('{app}\versions\{#VersionDir}');
end;

function CurrentPointer: AnsiString;
begin
  Result := '{"schema_version":1,"version_dir":"{#VersionDir}"}' + #13#10;
end;

procedure RequirePayloadFile(const Root, Name, ExpectedHash: String; AllowMissing: Boolean);
var
  Path: String;
begin
  Path := AddBackslash(Root) + Name;
  if AllowMissing and not FileExists(Path) then
    Exit;
  if not FileExists(Path) or (Lowercase(GetSHA256OfFile(Path)) <> Lowercase(ExpectedHash)) then
    RaiseException('Ferrum version file is missing or was changed: ' + Path + '. Existing versions have been preserved.');
end;

procedure VerifyVersionFiles(const Root: String; AllowMissing: Boolean);
begin
  RequirePayloadFile(Root, 'ferrum.exe', '{#GetSHA256OfFile(PayloadDir + "\ferrum.exe")}', AllowMissing);
  RequirePayloadFile(Root, 'ferrum-portable.json', '{#PayloadManifestHash}', AllowMissing);
  #define FindHandle
  #define FindResult
  #sub VerifyDll
    #define FoundName FindGetFileName(FindHandle)
    RequirePayloadFile(Root, '{#StringChange(FoundName, "'", "''")}', '{#GetSHA256OfFile(PayloadDir + "\" + FoundName)}', AllowMissing);
  #endsub
  #for {FindHandle = FindResult = FindFirst(PayloadDir + "\*.dll", 0); FindResult; FindResult = FindNext(FindHandle)} VerifyDll
  #if FindHandle
    #call FindClose(FindHandle)
  #endif
  #sub VerifyLicense
    #define FoundName FindGetFileName(FindHandle)
    RequirePayloadFile(Root, 'licenses\{#StringChange(FoundName, "'", "''")}', '{#GetSHA256OfFile(PayloadDir + "\licenses\" + FoundName)}', AllowMissing);
  #endsub
  #for {FindHandle = FindResult = FindFirst(PayloadDir + "\licenses\*", 0); FindResult; FindResult = FindNext(FindHandle)} VerifyLicense
  #if FindHandle
    #call FindClose(FindHandle)
  #endif
end;

procedure ActivateVersion;
var
  TemporaryPointer, ActivePointer: String;
  ResultCode: Integer;
begin
  VerifyVersionFiles(VersionPath, False);
  if not Exec(VersionPath + '\ferrum.exe', '--version', VersionPath,
      SW_HIDE, ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
    RaiseException('The new Ferrum version could not start. The previous active version has been preserved.');
  ActivePointer := ExpandConstant('{app}\current.json');
  // Reserve a fresh file beside current.json; an interrupted attempt cannot
  // leave a fixed staging name that blocks the next installation.
  SetLength(TemporaryPointer, 260);
  if GetTempFileName(ExpandConstant('{app}'), 'fer', 0, TemporaryPointer) = 0 then
    RaiseException('Cannot reserve a temporary Ferrum version pointer.');
  TemporaryPointer := Copy(TemporaryPointer, 1, Pos(#0, TemporaryPointer) - 1);
  // Publish only the small pointer. Running EXEs and DLLs remain in their version directory.
  try
    if not SaveStringToFile(TemporaryPointer, CurrentPointer, False) then
      RaiseException('Cannot prepare the Ferrum version switch.');
    // Keep ownership for every generated pointer. A failed activation can leave
    // the previous version active after Inno has updated its uninstall program.
    if not RegWriteStringValue(HKCU, VersionOwnershipKey, '{#VersionDir}', String(CurrentPointer)) then
      RaiseException('Cannot record Ferrum version ownership.');
    if not MoveFileEx(TemporaryPointer, ActivePointer, $1 or $8) then
      RaiseException('Cannot activate the new Ferrum version. The previous active version has been preserved.');
  finally
    // This invocation reserved the file, including any partial failed write.
    if FileExists(TemporaryPointer) then
      DeleteFile(TemporaryPointer);
  end;
end;

function OwnsPointer(const Contents: AnsiString): Boolean;
var
  Names: TArrayOfString;
  Recorded: String;
  Index: Integer;
begin
  Result := Contents = CurrentPointer;
  if Result or not RegKeyExists(HKCU, VersionOwnershipKey) then
    Exit;
  if not RegGetValueNames(HKCU, VersionOwnershipKey, Names) then
    RaiseException('Cannot read Ferrum version ownership.');
  for Index := 0 to GetArrayLength(Names) - 1 do begin
    if not RegQueryStringValue(HKCU, VersionOwnershipKey, Names[Index], Recorded) then
      RaiseException('Cannot read a Ferrum version ownership record.');
    if Recorded = String(Contents) then begin
      Result := True;
      Exit;
    end;
  end;
end;

function ExpandEnvironmentStrings(Source, Destination: String; Size: Cardinal): Cardinal;
  external 'ExpandEnvironmentStringsW@kernel32.dll stdcall';

// Inno 6 exposes pointer-sized INT_PTR/UINT_PTR, not HKEY or NativeInt.
// The script host's pointer width applies even in 64-bit installation mode.
function RegGetValue(Key: INT_PTR; SubKey, Name: String; Flags: Cardinal;
  var ValueType: Cardinal; Data: UINT_PTR; var Size: Cardinal): Longint;
  external 'RegGetValueW@advapi32.dll stdcall';

function UserPathExpandsVariables: Boolean;
var
  ValueType, Size: Cardinal;
  Status: Longint;
begin
  Size := 0;
  Status := RegGetValue(HKCU, EnvironmentKey, 'Path', StringTypesWithoutExpansion,
    ValueType, 0, Size);
  if Status = 2 then begin // ERROR_FILE_NOT_FOUND: no existing PATH value
    Result := False;
    Exit;
  end;
  if Status <> 0 then
    RaiseException('Cannot read the current-user PATH registry type (error ' + IntToStr(Status) + ').');
  Result := ValueType = RegistryExpandString;
end;

function ReadStringOrEmpty(const Key, Name: String): String;
begin
  Result := '';
  if RegValueExists(HKCU, Key, Name) then
    if not RegQueryStringValue(HKCU, Key, Name, Result) then
      RaiseException('Cannot read the current-user registry value ' + Key + '\' + Name + '.');
end;

function ComparablePath(Value: String; ExpandReferences: Boolean): String;
var
  Expanded: String;
  Needed, Written: Cardinal;
begin
  Value := Trim(Value);
  if Length(Value) >= 2 then
    if (Value[1] = '"') and (Value[Length(Value)] = '"') then
      Value := Copy(Value, 2, Length(Value) - 2);
  Expanded := Value;
  if ExpandReferences then begin
    Needed := ExpandEnvironmentStrings(Value, '', 0);
    if Needed = 0 then
      RaiseException('Cannot expand a current-user PATH entry.');
    Expanded := ''; // The Win32 input and output buffers must be distinct.
    SetLength(Expanded, Needed);
    Written := ExpandEnvironmentStrings(Value, Expanded, Needed);
    if (Written = 0) or (Written > Needed) then
      RaiseException('Cannot expand a current-user PATH entry.');
    SetLength(Expanded, Written - 1);
  end;
  StringChangeEx(Expanded, '/', '\', True);
  while Length(Expanded) > 3 do begin
    if Expanded[Length(Expanded)] <> '\' then
      Break;
    Delete(Expanded, Length(Expanded), 1);
  end;
  Result := Lowercase(Expanded);
end;

function FindPathEntry(const Value, Entry: String; Equivalent: Boolean;
  var First, Count: Integer): Boolean;
var
  Limit: Integer;
  Part: String;
  ExpandReferences: Boolean;
begin
  Result := False;
  ExpandReferences := False;
  if Equivalent then
    ExpandReferences := UserPathExpandsVariables;
  First := 1;
  while First <= Length(Value) do begin
    Limit := First;
    while Limit <= Length(Value) do begin
      if Value[Limit] = ';' then
        Break;
      Limit := Limit + 1;
    end;
    Count := Limit - First;
    Part := Copy(Value, First, Count);
    if Equivalent then
      Result := ComparablePath(Part, ExpandReferences) = ComparablePath(Entry, False)
    else
      Result := Part = Entry;
    if Result then
      Exit;
    First := Limit + 1;
  end;
end;

function OwnedEntry(const Suffix: String): String;
begin
  Result := Suffix;
  if Result <> '' then
    if Result[1] = ';' then
      Delete(Result, 1, 1);
end;

procedure ClearOwnership;
begin
  if RegValueExists(HKCU, OwnershipKey, OwnershipValue) then
    if not RegDeleteValue(HKCU, OwnershipKey, OwnershipValue) then
      RaiseException('Cannot remove Ferrum''s current-user PATH ownership record.');
  if RegValueExists(HKCU, OwnershipKey, PathExistedValue) then
    if not RegDeleteValue(HKCU, OwnershipKey, PathExistedValue) then
      RaiseException('Cannot remove Ferrum''s original PATH existence record.');
  RegDeleteKeyIfEmpty(HKCU, OwnershipKey);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  AppPath, Suffix, CurrentPath: String;
begin
  Result := '';
  try
    AppPath := ExpandConstant('{app}');
    if ComparablePath(AppPath, False) <>
       ComparablePath(ExpandConstant('{localappdata}\Programs\Ferrum'), False) then
      RaiseException('Ferrum must be installed in the current user''s LocalAppData\Programs\Ferrum directory.');
    if Pos(';', AppPath) <> 0 then
      RaiseException('The installation directory cannot contain a PATH separator (;).');
    RequireOrdinaryDirectory(AppPath);
    RequireOrdinaryDirectory(AppPath + '\versions');
    RequireOrdinaryDirectory(VersionPath);
    RequireOrdinaryDirectory(VersionPath + '\licenses');
    RequirePayloadFile(AppPath, 'ferrum.exe', '{#LauncherHash}', True);
    VerifyVersionFiles(VersionPath, True);
    CurrentPath := ReadStringOrEmpty(EnvironmentKey, 'Path');
    Suffix := ReadStringOrEmpty(OwnershipKey, OwnershipValue);
    if Suffix <> '' then
      if ComparablePath(OwnedEntry(Suffix), False) <> ComparablePath(AppPath, False) then
        RaiseException('The existing Ferrum PATH record belongs to another installation directory. Uninstall that installation first.');
  except
    Result := GetExceptionMessage;
  end;
end;

procedure AddUserPath;
var
  AppPath, CurrentPath, NewPath, Suffix, RollbackPath: String;
  First, Count: Integer;
  PathExisted, OwnershipSaved: Boolean;
  OriginalPathExisted: Cardinal;
begin
  AppPath := ExpandConstant('{app}');
  CurrentPath := ReadStringOrEmpty(EnvironmentKey, 'Path');
  Suffix := ReadStringOrEmpty(OwnershipKey, OwnershipValue);
  if Suffix <> '' then begin
    // A normal upgrade keeps the original ownership and adds nothing.
    if FindPathEntry(CurrentPath, OwnedEntry(Suffix), False, First, Count) then
      Exit;
    ClearOwnership;
  end;
  // Existing equivalent entries belong to the user, including quoted or expanded paths.
  if FindPathEntry(CurrentPath, AppPath, True, First, Count) then
    Exit;
  Suffix := AppPath;
  if CurrentPath <> '' then
    if CurrentPath[Length(CurrentPath)] <> ';' then
      Suffix := ';' + Suffix;
  NewPath := CurrentPath + Suffix;
  PathExisted := RegValueExists(HKCU, EnvironmentKey, 'Path');
  if PathExisted then OriginalPathExisted := 1 else OriginalPathExisted := 0;
  // RegWriteStringValue preserves an existing REG_EXPAND_SZ value's type.
  if not RegWriteStringValue(HKCU, EnvironmentKey, 'Path', NewPath) then
    RaiseException('Cannot add Ferrum to the current-user PATH.');
  OwnershipSaved := RegWriteStringValue(HKCU, OwnershipKey, OwnershipValue, Suffix);
  if OwnershipSaved then
    OwnershipSaved := RegWriteDWordValue(HKCU, OwnershipKey, PathExistedValue, OriginalPathExisted);
  if not OwnershipSaved then begin
    // Roll back only this unchanged write, never a later edit by another program.
    RollbackPath := ReadStringOrEmpty(EnvironmentKey, 'Path');
    if RollbackPath = NewPath then begin
      if PathExisted then
        RegWriteStringValue(HKCU, EnvironmentKey, 'Path', CurrentPath)
      else
        RegDeleteValue(HKCU, EnvironmentKey, 'Path');
    end;
    RaiseException('Cannot record Ferrum''s PATH ownership. Installation is incomplete.');
  end;
end;

procedure RemoveOwnedUserPath;
var
  CurrentPath, Suffix, Entry: String;
  First, Count: Integer;
  OriginalPathExisted: Cardinal;
  RemoveEmptyCreatedPath: Boolean;
begin
  Suffix := ReadStringOrEmpty(OwnershipKey, OwnershipValue);
  if Suffix = '' then
    Exit;
  Entry := OwnedEntry(Suffix);
  if ComparablePath(Entry, False) <> ComparablePath(ExpandConstant('{app}'), False) then
    RaiseException('Ferrum''s PATH ownership record does not match this installation.');
  // Older ownership records lack this field: preserve their PATH value conservatively.
  RemoveEmptyCreatedPath := False;
  if RegValueExists(HKCU, OwnershipKey, PathExistedValue) then begin
    if not RegQueryDWordValue(HKCU, OwnershipKey, PathExistedValue, OriginalPathExisted) then
      RaiseException('Cannot read Ferrum''s original PATH existence record.');
    if OriginalPathExisted > 1 then
      RaiseException('Ferrum''s original PATH existence record is invalid.');
    RemoveEmptyCreatedPath := OriginalPathExisted = 0;
  end;
  CurrentPath := ReadStringOrEmpty(EnvironmentKey, 'Path');
  // Remove at most one exact item. Never take ownership of a user-edited alias.
  if FindPathEntry(CurrentPath, Entry, False, First, Count) then begin
    if (First > 1) and (Suffix[1] = ';') then begin
      First := First - 1;
      Count := Count + 1;
    end else if First + Count <= Length(CurrentPath) then
      Count := Count + 1;
    Delete(CurrentPath, First, Count);
    if RemoveEmptyCreatedPath and (CurrentPath = '') then begin
      if not RegDeleteValue(HKCU, EnvironmentKey, 'Path') then
        RaiseException('Cannot remove the empty current-user PATH value created by Ferrum.');
    end else
      if not RegWriteStringValue(HKCU, EnvironmentKey, 'Path', CurrentPath) then
        RaiseException('Cannot remove Ferrum''s entry from the current-user PATH.');
  end;
  ClearOwnership;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then begin
    AddUserPath;
    ActivateVersion;
    UserPathConfigured := True;
  end;
end;

function GetCustomSetupExitCode: Integer;
begin
  Result := 0;
  // A post-install registry error must not become a successful silent setup.
  if not UserPathConfigured then
    Result := 1;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  PointerPath: String;
  Contents: AnsiString;
begin
  // Run before file deletion and Inno's ChangesEnvironment broadcast.
  // A registry error aborts before removing the installed files.
  if CurUninstallStep = usUninstall then begin
    RemoveOwnedUserPath;
    PointerPath := ExpandConstant('{app}\current.json');
    // Include an older managed pointer left active by an interrupted upgrade.
    if LoadStringFromFile(PointerPath, Contents) and OwnsPointer(Contents) then
      if not DeleteFile(PointerPath) then
        RaiseException('Cannot remove Ferrum''s active version pointer.');
    if RegKeyExists(HKCU, VersionOwnershipKey) then
      if not RegDeleteKeyIncludingSubkeys(HKCU, VersionOwnershipKey) then
        RaiseException('Cannot remove Ferrum version ownership.');
    RegDeleteKeyIfEmpty(HKCU, OwnershipKey);
  end;
end;
