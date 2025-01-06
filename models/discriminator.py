import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    El discriminador toma una imagen (real o generada) y produce
    un valor de probabilidad (real vs falso).
    """

    def __init__(self, nc=3, ndf=64):
        """
        Parámetros:
        ----------
        - nc: Número de canales de la imagen de entrada (3 para RGB).
        - ndf: Número base de filtros en el discriminador (64 es común en DCGAN).
        """
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # Bloque 1: (nc, 64, 64) -> (ndf, 32, 32)
            nn.Conv2d(in_channels=nc, out_channels=ndf,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Bloque 2: (ndf, 32, 32) -> (ndf*2, 16, 16)
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Bloque 3: (ndf*2, 16, 16) -> (ndf*4, 8, 8)
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Bloque 4: (ndf*4, 8, 8) -> (ndf*8, 4, 4)
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Bloque 5: (ndf*8, 4, 4) -> (1, 1, 1)
            nn.Conv2d(in_channels=ndf * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0, bias=False),
            
            # La salida es un tensor de tamaño [batch_size, 1, 1, 1].
            # Usualmente se pasa por una Sigmoid en el entrenamiento 
            # (depende de la versión de la GAN).
        )

    def forward(self, input):
        """
        input: tensor de forma (batch_size, nc, 64, 64)
        output: tensor (batch_size, 1, 1, 1) con la "probabilidad" de ser real.
        """
        return self.main(input)
