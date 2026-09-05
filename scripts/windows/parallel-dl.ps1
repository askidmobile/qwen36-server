# Параллельная загрузка одного большого файла с Hugging Face.
# Зачем: hf.exe на файлах ~20 ГБ вставал на 0 МБ/с (xet), одиночный curl
# деградировал до 1.5 МБ/с. N диапазонов дают полную полосу канала.
#
#   powershell -File parallel-dl.ps1 -Url <url> -OutFile <path> [-Parts 6]
#
# Докачка: существующий неполный OutFile становится part0 и дописывается.
# После склейки размер сверяется с content-length; при несовпадении части
# сохраняются для разбора (частая причина — старый файл длиннее диапазона
# part0, его нужно обрезать до размера диапазона).
param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$OutFile,
    [int]$Parts = 6
)
$ErrorActionPreference = 'Continue'
$dir = Split-Path -Parent $OutFile
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
$stem = Join-Path $dir ((Split-Path -Leaf $OutFile) + '.part')

$hdr = & curl.exe -sIL $Url
$clLine = ($hdr | Select-String -Pattern 'content-length' | Select-Object -Last 1).Line
$len = [int64]($clLine -replace '[^0-9]', '')
Write-Output ("content-length: {0} ({1:N1} GB)" -f $len, ($len / 1GB))
if ($len -lt 1) { Write-Output 'BAD content-length'; exit 1 }

$chunk = [int64][math]::Ceiling($len / $Parts)
if (Test-Path $OutFile) { Move-Item $OutFile "${stem}0" -Force }
$procs = @()
for ($i = 0; $i -lt $Parts; $i++) {
    $a = [int64]$i * $chunk
    $b = [int64][math]::Min(($i + 1) * $chunk - 1, $len - 1)
    $f = "$stem$i"
    $out = $f
    if ($i -eq 0 -and (Test-Path $f)) {
        $have = (Get-Item $f).Length
        if ($have -gt ($b - $a + 1)) {
            # Хвост прошлой загрузки длиннее диапазона — обрезать, иначе склейка съедет.
            $fs = [IO.File]::Open($f, 'Open', 'ReadWrite'); $fs.SetLength($b - $a + 1); $fs.Close()
            Write-Output 'part0: обрезан до размера диапазона'; continue
        }
        if ($have -eq ($b - $a + 1)) { Write-Output 'part0: готов'; continue }
        $a = $have; $out = "$f.tail"
    }
    $argList = @('-sS', '-L', '--retry', '30', '--retry-delay', '5', '--retry-all-errors', '-r', "$a-$b", $Url, '-o', $out)
    $procs += Start-Process -FilePath 'curl.exe' -ArgumentList $argList -NoNewWindow -PassThru -RedirectStandardError "$stem$i.err"
    Write-Output ("part{0}: {1}-{2}" -f $i, $a, $b)
}
$t0 = Get-Date
$s0 = (Get-ChildItem "$stem*" -Exclude '*.err' | Measure-Object Length -Sum).Sum
Start-Sleep -Seconds 60
$s1 = (Get-ChildItem "$stem*" -Exclude '*.err' | Measure-Object Length -Sum).Sum
Write-Output ("aggregate: {0:N1} MB/s" -f (($s1 - $s0) / 1MB / 60))
if ($procs.Count -gt 0) { $procs | Wait-Process }
Write-Output ("downloads done in {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds)

if (Test-Path "${stem}0.tail") {
    cmd /c "copy /b `"${stem}0`"+`"${stem}0.tail`" `"${stem}0.full`"" | Out-Null
    Move-Item "${stem}0.full" "${stem}0" -Force
    Remove-Item "${stem}0.tail" -Force
}
$list = ((0..($Parts - 1)) | ForEach-Object { '"' + $stem + $_ + '"' }) -join '+'
cmd /c ("copy /b " + $list + ' "' + $OutFile + '"') | Out-Null
$final = (Get-Item $OutFile).Length
Write-Output ("final: {0} of {1}" -f $final, $len)
if ($final -eq $len) {
    Remove-Item "$stem*" -Force
    Write-Output 'OK: части удалены'
} else {
    Write-Output 'SIZE MISMATCH: части оставлены для разбора'
    exit 1
}
