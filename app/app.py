import numpy as np
import torch
from PIL import Image
import urllib.request as request
from model import HeadgearRecognizer


def get_image_from_user() -> Image:
  while True:
    try:
      url = input('Enter image URL: ')
      image = Image.open(request.urlopen(url))
      return image
    except Exception as e:
      print('Invalid URL. Try again.')


def image_to_tensor(image: Image) -> torch.Tensor:
  image = image.convert('RGB')
  image = image.resize((224, 224))
  image = np.array(image)
  image = torch.tensor(image).to(torch.float32)
  image = image.permute(2, 0, 1)
  image = image.unsqueeze(0)
  return image


def detect_device() -> torch.device:
  if torch.cuda.is_available():
    device_type = 'cuda'
  elif torch.backends.mps.is_available():
    device_type = 'mps'
  else:
    device_type = 'cpu'
  return torch.device(device_type)


def load_model(state_path: str) -> torch.nn.Module:
  model = HeadgearRecognizer()
  device = detect_device()
  model.to(device)
  model_state = torch.load(state_path, map_location=device)
  model.load_state_dict(model_state)
  model.eval()
  return model


def main() -> None:
  model = load_model('model-0.8000.state')
  device = detect_device()
  index_to_class = {
    0: 'ASCOT CAP',
    1: 'BASEBALL CAP',
    2: 'BERET',
    3: 'BICORNE',
    4: 'BOATER',
    5: 'BOWLER',
    6: 'DEERSTALKER',
    7: 'FEDORA',
    8: 'FEZ',
    9: 'FOOTBALL HELMET',
    10: 'GARRISON CAP',
    11: 'HARD HAT',
    12: 'MILITARY HELMET',
    13: 'MOTARBOARD',
    14: 'PITH HELMET',
    15: 'PORK PIE',
    16: 'SOMBERO',
    17: 'SOUTHWESTER',
    18: 'TOP HAT',
    19: 'ZUCCHETTO'
  }
  while True:
    image = get_image_from_user()
    image = image_to_tensor(image).to(device)
    feedback = model(image)[0]

    # pick the top 5
    top5 = torch.topk(feedback, 5)
    for i in range(5):
      index = top5.indices[i].item()
      class_name = index_to_class[index]
      print(f'{i + 1}. {class_name}')
    print()
    
    # the answer
    answer = torch.argmax(feedback).item()
    answer = index_to_class[answer]
    print(f'Answer: {answer}')
    print()


if __name__ == '__main__':
  main()