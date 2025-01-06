import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm

# Importa tus módulos locales (asegúrate de que __init__.py exista en esas carpetas)
from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_loader import get_data_loader

def train_gan(
    epochs=5,
    batch_size=64,
    lr=0.0002,
    beta1=0.5,
    nz=100,
    resume_checkpoint=None  # <-- Parámetro opcional para reanudar
):
    """
    Entrena o reanuda el entrenamiento de una DCGAN.
    
    Parámetros:
    -----------
    epochs : int
        Número total de épocas a entrenar (o continuar).
    batch_size : int
        Tamaño de lote (batch) para el DataLoader.
    lr : float
        Tasa de aprendizaje (learning rate) para Adam.
    beta1 : float
        Parámetro beta1 de Adam (momento).
    nz : int
        Dimensión del vector de ruido para el generador.
    resume_checkpoint : str o None
        Ruta a un archivo .pth para reanudar entrenamiento. Si None, entrena desde cero.
    """

    # 1. Seleccionamos dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 2. DataLoader
    data_loader = get_data_loader(batch_size=batch_size)  # Carga CIFAR-10, normaliza, etc.

    # 3. Instanciamos modelos y optimizadores
    netG = Generator(nz=nz, ngf=64, nc=3).to(device)
    netD = Discriminator(nc=3, ndf=64).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    # Función de pérdida (BCE con logits)
    criterion = nn.BCEWithLogitsLoss()

    # Etiquetas para real y fake
    real_label = 1.0
    fake_label = 0.0

    # Vector de ruido fijo (para ver cómo evoluciona el generador)
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

    # 4. Si hay un checkpoint para reanudar
    start_epoch = 0
    if resume_checkpoint is not None:
        print(f"[*] Cargando checkpoint desde: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        # Cargamos estados de los modelos
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        
        # Cargamos estado de los optimizadores
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        
        # Recuperamos la época donde se quedó
        start_epoch = checkpoint['epoch'] + 1

        print(f"[*] Reanudando desde epoch {start_epoch}")

    # 5. Bucle de entrenamiento
    for epoch in range(start_epoch, epochs):
        for i, (images, _) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            
            # Mover imágenes al dispositivo
            images = images.to(device)

            ## ---------- A) Actualizar Discriminador ----------
            netD.zero_grad()

            # (A1) Entrenamiento con imágenes reales
            output_real = netD(images).view(-1)
            labels_real = torch.full((images.size(0),), real_label, device=device)
            lossD_real = criterion(output_real, labels_real)
            lossD_real.backward()

            # (A2) Entrenamiento con imágenes falsas
            noise = torch.randn(images.size(0), nz, 1, 1, device=device)
            fake_images = netG(noise)
            
            output_fake = netD(fake_images.detach()).view(-1)
            labels_fake = torch.full((images.size(0),), fake_label, device=device)
            lossD_fake = criterion(output_fake, labels_fake)
            lossD_fake.backward()

            # Actualizamos D
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            ## ---------- B) Actualizar Generador ----------
            netG.zero_grad()
            output_fake_for_G = netD(fake_images).view(-1)
            # El generador quiere que el D las califique como reales
            labels_real_for_G = torch.full((images.size(0),), real_label, device=device)
            lossG = criterion(output_fake_for_G, labels_real_for_G)
            lossG.backward()
            optimizerG.step()

            # Debug cada 100 iteraciones
            if i % 100 == 0:
                print(f"[Epoch {epoch+1}/{epochs} - Batch {i}/{len(data_loader)}] "
                      f"LossD: {lossD.item():.4f}, LossG: {lossG.item():.4f}")

        # Guardar algunas imágenes generadas al final de la época
        with torch.no_grad():
            fake_sample = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake_sample, f"outputs/generated_images/epoch_{epoch+1}.png", normalize=True)

        # 6. Guardar checkpoint al finalizar la época
        checkpoint_state = {
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict()
        }
        torch.save(checkpoint_state, f"outputs/checkpoints/checkpoint_epoch_{epoch+1}.pth")

        print(f"--- Epoch {epoch+1} finalizada. Checkpoint guardado. ---")


# ----- Punto de entrada -----
if __name__ == "__main__":
    # EJEMPLOS de uso:

    # EJEMPLO A: Entrenar desde cero 5 épocas
    # train_gan(epochs=5)

    # EJEMPLO B: Reanudar desde un checkpoint
    # (Imagina que guardaste uno en "outputs/checkpoints/checkpoint_epoch_3.pth")
    train_gan(epochs=15, resume_checkpoint="outputs/checkpoints/checkpoint_epoch_5.pth")

    # Puedes comentar/descomentar según lo necesites:
    # train_gan(epochs=15, batch_size=64, lr=0.0002, beta1=0.5, nz=100)
