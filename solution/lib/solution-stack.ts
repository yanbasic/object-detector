import * as cdk from '@aws-cdk/core';
import * as agw from '@aws-cdk/aws-apigateway';
import * as lambda from '@aws-cdk/aws-lambda';


export class OcrWithDetectorSolutionStack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
    this.templateOptions.description = `(SO8023-ocr) - AI Solution Kits - Infer OCR with Object Detector. Template version v1.0.0`;
    /**
     * Lambda Provision
     */
    const factory2DCodeRecognizer = new lambda.DockerImageFunction(
        this,
        'factory2DCodeRecognizer',
        {
            code: lambda.DockerImageCode.fromImageAsset('./lambda/pcbLogoDet'),
            timeout: cdk.Duration.seconds(15),
            memorySize: 10240,
        }
    );

    /**
     * API Gateway Provision
     */
    const factoryApiRouter = new agw.RestApi(
        this,
        'OcrDetectorApiRouter',
        {
            endpointConfiguration: {
                types: [agw.EndpointType.REGIONAL]
            },
            defaultCorsPreflightOptions: {
                allowOrigins: agw.Cors.ALL_ORIGINS,
                allowMethods: agw.Cors.ALL_METHODS
            }
        }
    );
    factoryApiRouter.root.addResource('detect').addMethod('POST', new agw.LambdaIntegration(factory2DCodeRecognizer),
    {
        authorizationType: agw.AuthorizationType.IAM
    });



  }
}
