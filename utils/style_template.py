style_list = [
    {
        "name": "No_style",
        "prompt": "{prompt} 8k,RAW",
        "negative_prompt": "worst quality",
    },
    {
        "name": "Realistic",
        "prompt": "{prompt}  (photorealistic:1.4),8k,RAW photo, best quality, masterpiece,dramatic shadows,uhd, high quality,dramatic,cinematic",
        "negative_prompt": "(worst quality, low quality, normal quality, lowres), (bad teeth, deformed teeth, deformed lips), (bad proportions), (deformed eyes, bad eyes), (deformed face, ugly face), (deformed hands,fused fingers), morbid, mutilated, mutation, disfigured",
    },
    {
        "name": "Japanese_Anime",
        "prompt": "{prompt} (Anime Style, Manga Style:1.3)anime artwork illustrating,best quality,webtoon,manhua,Linear compositions",
        "negative_prompt": "lowres,text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
        "name": "Digital_Oil_Painting",
        "prompt": "{prompt}  (Extremely Detailed Oil Painting:1.2), glow effects, godrays,render, 8k, octane render, cinema 4d, blender, dark, atmospheric 4k ultra detailed, cinematic sensual,humorous illustration ",
        "negative_prompt": "anime, cartoon, graphic, text, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, lowres, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
        "name": "Pixar_Disney_Character",
        "prompt": "Create a Disney Pixar 3D style illustration on {prompt}  The scene is vibrant, motivational, filled with vivid colors and a sense of wonder.",
        "negative_prompt": "lowres, text, bad eyes, bad legs, error,extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry, grayscale, noisy, sloppy, messy, grainy,ultra textured,",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt}  Hyperrealistic, Hyperdetailed, detailed skin,soft lighting, best quality, ultra realistic, 8k, golden ratio, Intricate, High Detail, film photography",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, lowres, text, error,  extra digit, fewer digits, cropped, worst quality, low quality,signature, watermark, username",
    },
    {
        "name": "Comic_book",
        "prompt": "comic {prompt} graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
        "negative_prompt": "photograph, deformed, glitch, noisy, realistic, stock photo, lowres,text, error,extra digit, fewer digits, cropped, worst quality, low quality, normal quality,signature, watermark, username, blurry",
    },
    {
        "name": "Line_art",
        "prompt": "line art drawing {prompt}  professional, sleek, modern, minimalist, graphic, line art, vector graphics",
        "negative_prompt": "photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes,ugly, disfigured, mutated, realism, realistic, impressionism, expressionism,acrylic, lowres, text, error,extra digit, fewer digits,  worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username",
    },
    {
        "name": "Black_and_White_Film_Noir",
        "prompt": "{prompt}  (b&w, Monochromatic, Film Photography:1.3), film noir, analog style, soft lighting, subsurface scattering,heavy shadow, masterpiece, best quality,",
        "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes,ugly, disfigured, mutated, impressionism, expressionism,acrylic, lowres,text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username",
    },
    {
        "name": "Isometric_Rooms",
        "prompt": "Tiny cute isometric {prompt} in a cutaway box, soft smooth lighting, soft colors, 100mm lens, 3d blender render",
        "negative_prompt": "photorealistic,35mm film,deformed,glitch,blurry,noisy,off-center,deformed,cross-eyed,closed eyes,ugly,disfigured,mutated,realism,realistic,impressionism,expressionism,acrylic,lowres,text,error,extra digit,fewer digits,cropped, worst quality, low quality, normal quality, jpeg artifacts, signature,watermark, username,",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
