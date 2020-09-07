import argparse

import torch

def get_args_vanilla():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--num-envs',
        type=int,
        default=72,
        help='how many training CPU processes to use (default: 72)')
    parser.add_argument(
        '--start-level',
        type=int,
        default=0,
        help='general seed (default: 0)')
    parser.add_argument(
        '--num-levels',
        type=int,
        default=0,
        help='how many levels in one env, if 0 no limit. set this 1 to fix env state(background) (default: 0)')
    parser.add_argument(
        '--env-name',
        default='starpilot',
        help='environment to train on (default: starpilot)')
    parser.add_argument(
        '--log-dir',
        default='./logs/',
        help='directory to save agent logs (default: ./logs/')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')

    args = parser.parse_args()
    return args

def get_args_continual_baselines():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--start-level-min',
        type=int,
        default=0,
        help='general seed min (default: 0)')
    parser.add_argument(
        '--start-level-max',
        type=int,
        default=5,
        help='general seed max (default: 5)')
    parser.add_argument(
        '--num-envs',
        type=int,
        default=72,
        help='how many training CPU processes to use (default: 72)')
    parser.add_argument(
        '--env-name',
        default='starpilot',
        help='environment to train on (default: starpilot)')
    parser.add_argument(
        '--log-dir',
        default='./logs/',
        help='directory to save agent logs (default: ./logs/')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')

    args = parser.parse_args()
    return args

def get_args_continual():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--start-level-min',
        type=int,
        default=0,
        help='general seed min (default: 0)')
    parser.add_argument(
        '--start-level-max',
        type=int,
        default=5,
        help='general seed max (default: 5)')
    parser.add_argument(
        '--num-envs',
        type=int,
        default=72,
        help='how many training CPU processes to use (default: 72)')
    parser.add_argument(
        '--env-name',
        default='starpilot',
        help='environment to train on (default: starpilot)')
    parser.add_argument(
        '--log-dir',
        default='./logs/',
        help='directory to save agent logs (default: ./logs/')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    
    parser.add_argument(
        '--cam-loss-coef',
        type=float,
        default=1.0,
        help='cam-loss-coef (default: 1)')
    parser.add_argument(
        "--logit-loss-coef", 
        type=float,
        default=1.0,
        help='logit-loss-coef" (default: 1)')
    parser.add_argument(
        "--value-loss-coef", 
        type=float,
        default=1.0,
        help='value-loss-coef" (default: 1)')

    args = parser.parse_args()
    return args