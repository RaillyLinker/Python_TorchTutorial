# L1 정규화 (L1 Regularization):
# L1 정규화는 가중치(weight)의 절대값을 손실 함수에 추가함으로써 모델의 복잡성을 제어합니다.
# 손실 함수에 가중치의 L1 노름(norm)을 추가합니다. 이는 가중치의 각 요소의 절대값의 합으로 정의됩니다.
# L1 정규화는 가중치를 희소하게 만들어 모델의 복잡성을 줄입니다. 즉, 일부 가중치를 0으로 만들어 특성 선택(feature selection)과 같은 효과를 줄 수 있습니다.
# 예를 들어, L1 정규화를 사용하면 Logistic 회귀의 Lasso 모델이라고도 불리는 희소한 모델을 얻을 수 있습니다.
def l1_regularization(train_dataloader, device, model, criterion):
    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)

        # L1 Norm 적용 비율 (0.001 같은 작은 값을 사용하는 것이 일반적입니다.)
        _lambda = 0.5
        # L1 Norm : 모델 파라미터에 abs 적용 후 더하기
        l1_norm = sum(p.abs().sum() for p in model.parameters())

        # L1 정규화를 적용하려면, l1_norm * _lambda 값을 손실값에 더해주면 됩니다.
        loss = criterion(output, y) + _lambda * l1_norm


# L2 정규화 (L2 Regularization):
# L2 정규화는 가중치의 제곱항을 손실 함수에 추가하여 모델의 복잡성을 제어합니다.
# 손실 함수에 가중치의 L2 노름(norm)의 제곱을 추가합니다. 이는 가중치의 각 요소의 제곱의 합으로 정의됩니다.
# L2 정규화는 가중치를 작게 만들어 모델의 파라미터 값들이 작게 유지되도록 합니다. 이는 모델의 일반화를 향상시키는 효과가 있습니다.
# 예를 들어, L2 정규화는 Ridge 회귀와 같은 모델에서 사용되며, 이는 모든 특성들이 모델에 동시에 영향을 주도록 합니다.
def l2_regularization(train_dataloader, device, model, criterion):
    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)

        # L2 Norm 적용 비율 (0.001 같은 작은 값을 사용하는 것이 일반적입니다.)
        _lambda = 0.5
        # L2 Norm : 모델 파라미터를 제곱 후 더하기
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

        # L2 정규화를 적용하려면, l2_norm * _lambda 값을 손실값에 더해주면 됩니다.
        loss = criterion(output, y) + _lambda * l2_norm

# 일반적으로 L2 정규화가 L1 정규화보다 더 많이 사용되며,
# 가중치를 작게 유지하면서도 희소성의 효과가 없으므로 모델의 일반화를 향상시키는데 효과적입니다.
# 그러나 데이터셋이 매우 크거나 특성이 매우 많은 경우에는 L1 정규화가 희소한 모델을 얻는 데 유용할 수 있습니다.
# 종종 L1 정규화와 L2 정규화를 함께 사용하여 모델의 성능을 향상시키는 경우도 있습니다.
