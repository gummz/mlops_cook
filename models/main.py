import argparse
import sys
import tqdm
import matplotlib.pyplot as plt

import torch

from data import mnist
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.01)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()

        train_set, _ = mnist()
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
        
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        epochs = 30
        train_losses = []

        for epoch in tqdm.tqdm(range(epochs),unit='epoch'):
            running_loss = 0
            for images,labels in train_loader:
                
                # zero grad
                optimizer.zero_grad()
                
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                # backward step
                loss.backward()
                # opt step
                optimizer.step()
                
                running_loss += loss.item()
            train_losses.append(running_loss)

        torch.save(model.state_dict(), 'trained_model.pth')
        plt.plot(train_losses)
        plt.show()



        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        _, test_set = mnist()
        testloader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)
        res = []

        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                log_ps = model(images)

                # get probabilities
                ps = torch.exp(log_ps)

                _, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                # get results
                res.append(equals)

            equals = torch.cat(res)
            accuracy = torch.mean(equals.type(torch.FloatTensor))

            print(f'The accuracy of the model is: { accuracy.item() * 100 }%')


if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    