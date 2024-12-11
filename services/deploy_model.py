from model_deployer import ModelDeployment


def deploy_model():
    """Deploy the model"""
    deployer = ModelDeployment(environment="local", project = "pk-arg-prj4-datasci",location = "us-central1")
    model_name = "credit_card_prediction_model"
    deployer.upload_model_sample(display_name=model_name)
    deployer.deploy_model(model_name)


if __name__ == "__main__":
    deploy_model()
