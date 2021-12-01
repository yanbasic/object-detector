import * as cdk from '@aws-cdk/core';
import * as agw from '@aws-cdk/aws-apigateway';
import * as lambda from '@aws-cdk/aws-lambda';


export class SolutionStack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    /**
     * Lambda Provision
     */
    const factory2DCodeRecognizer = new lambda.DockerImageFunction(
        this,
        'northStarLoginAuth',
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
        'factoryApiRouter',
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
    factoryApiRouter.root.addResource('detect').addMethod('POST', new agw.LambdaIntegration(factory2DCodeRecognizer));



  }
}
