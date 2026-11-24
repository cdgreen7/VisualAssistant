"""Offline image handling; originals never inherit production upload limits."""
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
EXTENSIONS={'.jpg','.jpeg','.png','.heic','.heif'}


def resolve(path):
    p=Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def discover(folder):
    root=resolve(folder)
    if not root.exists():raise FileNotFoundError(root)
    paths=[root] if root.is_file() else root.rglob('*')
    return sorted({p.resolve() for p in paths if p.is_file() and p.suffix.lower() in EXTENSIONS})


def heic_pixels(path):
    """Decode a HEIC/HEIF file to JPEG bytes with macOS's built-in `sips`.

    Nothing is installed and the original is not touched: sips writes a
    temporary copy, which is read back and deleted.
    """
    tool=shutil.which('sips')
    if not tool:raise ValueError('HEIC decoder unavailable; use an existing converted JPEG or install a decoder separately')
    with tempfile.TemporaryDirectory() as folder:
        out=Path(folder)/'decoded.jpg'
        try:
            done=subprocess.run([tool,'-s','format','jpeg','-s','formatOptions','best',str(path),'--out',str(out)],
                                capture_output=True,timeout=120)
        except (OSError,subprocess.SubprocessError) as error:raise ValueError(f'HEIC decoding failed: {type(error).__name__}') from None
        if done.returncode or not out.exists():raise ValueError('HEIC decoding failed; the file may be damaged')
        return out.read_bytes()


def read(path):
    path=Path(path)
    max_bytes=int(os.getenv('VA_RESEARCH_MAX_BYTES',100*1024*1024))
    max_pixels=int(os.getenv('VA_RESEARCH_MAX_PIXELS',60_000_000))
    if path.stat().st_size>max_bytes:raise ValueError('Offline byte safety limit exceeded')
    data=path.read_bytes()
    if len(data)>max_bytes:raise ValueError('Offline byte safety limit exceeded')
    heic=path.suffix.lower() in ('.heic','.heif');pixels=data;decoder='Pillow'
    if heic:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener();decoder='pillow_heif'
        except ImportError:
            # The hash below is always of the ORIGINAL bytes; only pixels come from the decoded copy.
            pixels=heic_pixels(path);decoder='macOS sips (temporary JPEG copy)'
    with Image.open(io.BytesIO(pixels)) as source:
        if source.format not in ('JPEG','PNG','HEIF','HEIC'):raise ValueError('Only JPEG and PNG (or decodable HEIC) supported')
        if source.width*source.height>max_pixels:raise ValueError('Offline pixel safety limit exceeded')
        exif=source.getexif();orientation=exif.get(274,1)
        info={'sha256':hashlib.sha256(data).hexdigest(),
              'extension':'.heic' if heic else {'JPEG':'.jpg','PNG':'.png'}[source.format],
              'width':source.width,'height':source.height,'has_exif':bool(exif),'orientation':orientation,'decoder':decoder}
        upright=ImageOps.exif_transpose(source).convert('RGB')
    return data,info,upright


def image_info(path):
    data,info,image=read(path);image.close();return data,info


def fingerprint(image):
    thumb=image.resize((9,8)).convert('L');pixels=list(thumb.getdata());bits=0
    for y in range(8):
        for x in range(8):bits=(bits<<1)|int(pixels[y*9+x]>pixels[y*9+x+1])
    color=image.resize((1,1)).getpixel((0,0))
    return {'dhash':f'{bits:016x}','aspect':image.width/image.height,'color':list(color)}


def related(a,b):
    if Path(a['source']).stem.casefold()==Path(b['source']).stem.casefold():return 'same filename stem (weak metadata)'
    x,y=a.get('fingerprint'),b.get('fingerprint')
    if not x or not y:return None
    if abs(x['aspect']/y['aspect']-1)<.03 and sum(abs(i-j) for i,j in zip(x['color'],y['color']))<90:
        if bin(int(x['dhash'],16)^int(y['dhash'],16)).count('1')<=6:return 'similar appearance; may be a different photograph/product'
    return None


def derived(data_image,max_side=2560):
    """Separate, bounded endpoint input. EXIF transpose happened in read()."""
    image=data_image.copy();image.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    while True:
        out=io.BytesIO();image.save(out,'JPEG',quality=92)
        if len(out.getvalue())<=8*1024*1024 and image.width*image.height<=7_000_000:break
        image.thumbnail((max(1,int(image.width*.8)),max(1,int(image.height*.8))),Image.Resampling.LANCZOS)
    return out.getvalue(),{'source_width':data_image.width,'source_height':data_image.height,'orientation':'EXIF transpose','resize':'aspect-preserving LANCZOS, no upscale',
                         'crop':None,'width':image.width,'height':image.height,'jpeg_quality':92}
