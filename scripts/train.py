import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm  # para barra de progreso

# Importamos nuestras clases y el DataLoader personalizado
from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_loader import get_data_loader  # Ajusta la ruta según tu estructura

def train_gan(epochs=5, batch_size=64, lr=0.0002, beta1=0.5, nz=100):
    """
    Entrena una DCGAN con los modelos Generator y Discriminator definidos.
    
    Parámetros:
    -----------
    epochs : int
        Número de iteraciones completas sobre el dataset.
    batch_size : int
        Tamaño de lote usado en el entrenamiento.
    lr : float
        Tasa de aprendizaje para Adam.
    beta1 : float
        Parámetro beta1 para Adam (momento exponencial).
    nz : int
        Dimensionalidad del vector de ruido.
    """
    
    # 1. Preparar dispositivo (GPU si está disponible, si no CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 2. Obtener DataLoader
    data_loader = get_data_loader(batch_size=batch_size)

    # 3. Instanciar el generador (G) y el discriminador (D)
    netG = Generator(nz=nz, ngf=64, nc=3).to(device)
    netD = Discriminator(nc=3, ndf=64).to(device)

    # 4. Definir función de pérdida (BCE, por ejemplo) y optimizadores (Adam)
    criterion = nn.BCEWithLogitsLoss()  # A menudo se usa sin Sigmoid final en Discriminator
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    # Etiquetas para real y fake (para la pérdida)
    real_label = 1.0
    fake_label = 0.0

    # Vector de ruido fijo (para ver evolución de imágenes)
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

    # 5. Bucle de entrenamiento
    for epoch in range(epochs):
        # tqdm nos da una barra de progreso por cada epoch
        for i, (images, _) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Mover imágenes al dispositivo
            images = images.to(device)

            ## ------------------ (A) Actualizar Discriminador ------------------
            netD.zero_grad()

            # (A1) Entrenamiento con imágenes reales
            # Salida del D para las imágenes reales
            output_real = netD(images).view(-1)
            # Creamos etiqueta real
            labels_real = torch.full((images.size(0),), real_label, dtype=torch.float, device=device)
            # Calculamos pérdida
            lossD_real = criterion(output_real, labels_real)
            # Hacemos backprop
            lossD_real.backward()

            # (A2) Entrenamiento con imágenes falsas generadas
            noise = torch.randn(images.size(0), nz, 1, 1, device=device)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach()).view(-1)
            labels_fake = torch.full((images.size(0),), fake_label, dtype=torch.float, device=device)
            lossD_fake = criterion(output_fake, labels_fake)
            lossD_fake.backward()

            # Sumamos las pérdidas
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            ## ------------------ (B) Actualizar Generador ------------------
            netG.zero_grad()
            # El generador quiere que el discriminador marque sus muestras como reales
            output_fake_for_G = netD(fake_images).view(-1)
            labels_real_for_G = torch.full((images.size(0),), real_label, dtype=torch.float, device=device)
            lossG = criterion(output_fake_for_G, labels_real_for_G)
            lossG.backward()
            optimizerG.step()

            ## (Opcional) Cada X iteraciones, puedes imprimir o guardar las pérdidas
            if i % 100 == 0:
                print(f"[Epoch {epoch+1}/{epochs} - Batch {i}/{len(data_loader)}] "
                      f"LossD: {lossD.item():.4f}, LossG: {lossG.item():.4f}")

        # Guardar imágenes de ejemplo al final de cada epoch
        with torch.no_grad():
            fake_sample = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake_sample, f"outputs/generated_images/epoch_{epoch+1}.png", normalize=True)

        # (Opcional) Guardar checkpoints
        torch.save(netG.state_dict(), f"outputs/checkpoints/netG_epoch_{epoch+1}.pth")
        torch.save(netD.state_dict(), f"outputs/checkpoints/netD_epoch_{epoch+1}.pth")

    print("Entrenamiento completado.")

if __name__ == "__main__":
    # Llamar a la función de entrenamiento
    train_gan(epochs=5, batch_size=64, lr=0.0002, beta1=0.5, nz=100)
