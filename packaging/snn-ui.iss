; Inno Setup script — wrap dist/snn-ui.exe into a Windows installer (setup.exe) with
; Start-Menu and optional desktop shortcuts.
;
; 1. Build the exe first:      python packaging/build_exe.py     -> dist/snn-ui.exe
; 2. Compile this installer with Inno Setup 6 (https://jrsoftware.org/isinfo.php):
;      "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\snn-ui.iss
;    -> packaging/installer/snn-ui-setup.exe
;
; The .exe is already portable/double-click-runnable on its own; this installer is only for
; a Start-Menu entry + uninstaller.

[Setup]
AppName=Spiking Neural Data Lake
AppVersion=2.53.2
AppPublisher=Aighluvsekks
DefaultDirName={autopf}\SpikingNeuralDataLake
DefaultGroupName=Spiking Neural Data Lake
DisableProgramGroupPage=yes
OutputDir=installer
OutputBaseFilename=snn-ui-setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\snn-ui.exe

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "..\dist\snn-ui.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Spiking Neural Data Lake"; Filename: "{app}\snn-ui.exe"
Name: "{group}\Uninstall Spiking Neural Data Lake"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Spiking Neural Data Lake"; Filename: "{app}\snn-ui.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\snn-ui.exe"; Description: "Launch Spiking Neural Data Lake"; Flags: nowait postinstall skipifsilent
