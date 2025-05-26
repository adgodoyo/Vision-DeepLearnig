from yt_dlp import YoutubeDL       # Importa la clase YoutubeDL de yt-dlp para descargar videos
from pathlib import Path           # Importa Path para manejo de rutas de forma segura

# Lista de URLs de videos de TikTok a descargar
urls = [
    "https://www.tiktok.com/@rcnradiocolombia/video/7266222798954138886",
    "https://www.tiktok.com/@seguridad_bogota/video/7453973210762169606",
    "https://www.tiktok.com/@seguridad_bogota/video/7443574213056777527",
    "https://www.tiktok.com/@j0hnrt9/video/7487425109532232966",
    "https://www.tiktok.com/@alertabogota104.4/video/7431627870595828997",
    "https://www.tiktok.com/@valeg_.29/video/7506718369387728134",
    "https://www.tiktok.com/@steve.r.m/video/7231951922595515654",
    "https://www.tiktok.com/@isabelag_08/video/7434558222985366839",
    "https://www.tiktok.com/@herli1999/video/7296254514749918469",
    "https://www.tiktok.com/@samuuuu.ell/video/7421344986647612678",
    "https://www.tiktok.com/@redmasnoticias/video/7506895468740283654",
    "https://www.tiktok.com/@angee_j01/video/7468414447468449030",
    "https://www.tiktok.com/@jhonguantivajoya/video/7168633569571605765",
    "https://www.tiktok.com/@saul.y/video/7289254458821856517",
    "https://www.tiktok.com/@noticiascaracol/video/7366258598819089670",
    "https://www.tiktok.com/@gerardosarmiento_veedoru/video/7238753432134225157"
]

def descargar_videos(video_urls, output_dir="data/raw"):
    # Crea el directorio de salida si no existe (incluye padres)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Opciones de configuración para yt-dlp
    opciones = {
        'format': 'mp4',                                    # Formato de salida MP4
        'outtmpl': f'{output_dir}/video_%(id)s.%(ext)s',    # Plantilla de nombre de archivo
    }
    # Inicia YoutubeDL con las opciones definidas y descarga cada URL
    with YoutubeDL(opciones) as ydl:
        ydl.download(video_urls)
