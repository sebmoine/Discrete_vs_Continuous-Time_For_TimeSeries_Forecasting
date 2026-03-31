from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    

def get_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, params):
    BATCH_SIZE, NUM_WORKERS = params

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True if NUM_WORKERS > 0 else False)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True if NUM_WORKERS > 0 else False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True if NUM_WORKERS > 0 else False)

    # for input ,output in train_loader:
    #     x_batch , y_batch = input.to(device), output.to(device)
    #     print("For TRAIN :", x_batch.shape, y_batch.shape)
    #     break
    # for input ,output in test_loader:
    #     x_batch , y_batch = input.to(device), output.to(device)
    #     print("For TEST :", x_batch.shape, y_batch.shape)
    #     break

    return train_loader, val_loader, test_loader