import kagglehub

# Download latest version
path = kagglehub.competition_download("enveda-CASMI26-molecule-id-mass-spectra")

print("Path to competition files:", path)
