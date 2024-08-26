from PIL import Image

def unisci_immagini_orizzontalmente(img1_path, img2_path, img3_path, img4_path, output_path):
    # Apre le immagini
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img3 = Image.open(img3_path)
    img4 = Image.open(img4_path)
    
    # Combina le immagini orizzontalmente
    larghezza_totale = img1.width + img2.width + img3.width + img4.width
    altezza_massima = max(img1.height, img2.height, img3.height, img4.height)
    
    nuova_immagine = Image.new('RGB', (larghezza_totale, altezza_massima))
    nuova_immagine.paste(img1, (0, 0))
    nuova_immagine.paste(img2, (img1.width, 0))
    nuova_immagine.paste(img3, (img1.width + img2.width, 0))
    nuova_immagine.paste(img4, (img1.width + img2.width + img3.width, 0))
    
    # Salva la nuova immagine
    nuova_immagine.save(output_path)

# Utilizzo della funzione per unire tre immagini orizzontalmente
unisci_immagini_orizzontalmente("./imgs/plot_umap_joint_emotion_cosine.png"
                                , "./imgs/plot_umap_bone_emotion_cosine.png"
                                , "./imgs/plot_umap_joint_motion_emotion_cosine.png"
                                ,"./imgs/plot_umap_bone_motion_emotion_cosine.png"
                                , "./imgs/immagine_combinata_space_cosine.png"
                                )
