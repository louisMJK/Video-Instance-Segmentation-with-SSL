from large_model import LargeModel
from data import ImagesToMaskDataset



device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'\nTraining on {device}.')


hidden_dataset = ImagesToMaskDataset(root='../../dataset/hidden/')
hidden_loader = torch.utils.data.DataLoader(hidden_dataset, batch_size=1, shuffle=False, num_workers=1)

model = LargeModel()
#TODO: Load the rest of the model
model.predictor.load_state_dict(torch.load('predictor_model.pth'))

model.to(device)

model.eval()
list_of_results = []
for x, _ in hidden_loader:
    x = x.to(device)
    with torch.no_grad():
        y = model(x)
    y = y.cpu().numpy()
    assert y.shape == (160, 240)
    list_of_results.append(y)

results = np.array(list_of_results)

#TODO: Save


    
