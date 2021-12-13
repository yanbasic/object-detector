#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from '@aws-cdk/core';
import {BootstraplessStackSynthesizer} from 'cdk-bootstrapless-synthesizer';
import { OcrWithDetectorSolutionStack } from '../lib/solution-stack';

const app = new cdk.App();
new OcrWithDetectorSolutionStack(app, 'OcrWithDetectorSolutionStack', {synthesizer: synthesizer()});

app.synth()

function synthesizer() {
    return process.env.USE_BSS ? new BootstraplessStackSynthesizer() : undefined;
}