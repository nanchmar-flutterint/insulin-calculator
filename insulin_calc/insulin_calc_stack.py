import subprocess

from aws_cdk import (
    Stack,
    aws_dynamodb as dynamodb,
    aws_lambda as lambda_,
    aws_apigatewayv2 as apigwv2,
    aws_iam as iam,
    aws_secretsmanager as secretsmanager,
    aws_logs as logs,
    CfnOutput as CfnOutput,
    Duration,
    BundlingOptions,
    aws_events,
    aws_events_targets,
)
from aws_cdk.aws_iam import PolicyDocument
from aws_cdk.aws_scheduler import Schedule, ScheduleExpression
from constructs import Construct
import json
import os


class InsulinCalcStack(Stack):
    MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"

    def create_dependencies_layer_cgm(
        self, project_name, function_name: str
    ) -> lambda_.LayerVersion:
        requirements_file = "lambda/cgm/" + function_name + ".txt"
        output_dir = ".lambda/cgm/" + function_name

        # Install requirements for layer in the output_dir
        if not os.environ.get("SKIP_PIP"):
            # Note: Pip will create the output dir if it does not exist
            subprocess.check_call(
                f"pip install -r {requirements_file} -t {output_dir}/python".split()
            )
        return lambda_.LayerVersion(
            self,
            project_name + "-" + function_name + "-dependencies",
            code=lambda_.Code.from_asset(output_dir),
        )

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # The code that defines your stack goes here

        ##########################################################################
        #   Getting Account ID and Region as variables                           #
        ##########################################################################

        self._accountId = os.environ["CDK_DEFAULT_ACCOUNT"]
        self._region = os.environ["CDK_DEFAULT_REGION"]

        ##########################################################################
        #   Log Group                                                           #
        ##########################################################################

        # Configure log group for short retention
        logGroup = logs.CfnLogGroup(
            self,
            "LogGroup",
            log_group_name="/api/requests/NutritionHistory",
            retention_in_days=14,
        )

        ##########################################################################
        #   Dynamo DB                                                            #
        ##########################################################################
        table_name = "NutritionHistory"
        # Dynamo DB
        table = dynamodb.Table(
            self,
            "EventTable",
            partition_key=dynamodb.Attribute(
                name="DeviceId", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="Date", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            table_name=table_name,
            time_to_live_attribute="TTL",
            resource_policy=PolicyDocument(
                statements=[
                    iam.PolicyStatement(
                        actions=[
                            "dynamodb:DescribeTable",
                            "dynamodb:GetItem",
                            "dynamodb:PutItem",
                            "dynamodb:DeleteItem",
                            "dynamodb:UpdateItem",
                            "dynamodb:Query",
                            "dynamodb:Scan",
                        ],
                        resources=[
                            f"arn:aws:dynamodb:{self._region}:{self._accountId}:table/{table_name}"
                        ],
                        principals=[iam.AccountRootPrincipal()],
                        conditions={
                            "StringEquals": {
                                "aws:PrincipalArn": f"arn:aws:iam::{self._accountId}:root"
                            }
                        },
                    )
                ]
            ),
        )

        table_name2 = "CGM"
        # Dynamo DB
        table2 = dynamodb.Table(
            self,
            "CGM",
            partition_key=dynamodb.Attribute(
                name="DeviceId", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="Date", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            table_name=table_name2,
            time_to_live_attribute="TTL",
            resource_policy=PolicyDocument(
                statements=[
                    iam.PolicyStatement(
                        actions=[
                            "dynamodb:DescribeTable",
                            "dynamodb:GetItem",
                            "dynamodb:PutItem",
                            "dynamodb:DeleteItem",
                            "dynamodb:UpdateItem",
                            "dynamodb:Query",
                            "dynamodb:Scan",
                        ],
                        resources=[
                            f"arn:aws:dynamodb:{self._region}:{self._accountId}:table/{table_name}"
                        ],
                        principals=[iam.AccountRootPrincipal()],
                        conditions={
                            "StringEquals": {
                                "aws:PrincipalArn": f"arn:aws:iam::{self._accountId}:root"
                            }
                        },
                    )
                ]
            ),
        )

        ##########################################################################
        #   Secrets - Generated for Endpoint Auth                                #
        ##########################################################################

        # Secret for username
        secretUsername = secretsmanager.Secret(
            self,
            "authUsername",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template=json.dumps({}),
                generate_string_key="username",
                exclude_punctuation=True,
            ),
        )

        # Secret for password
        secretPassword = secretsmanager.Secret(
            self,
            "authPassword",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template=json.dumps({}),
                generate_string_key="password",
                exclude_punctuation=True,
            ),
        )

        ##########################################################################
        #   Roles                                                                #
        ##########################################################################

        # API GW to SQS Role
        ApiGwToSqsRole = iam.Role(
            self,
            "ApiGwV2ToSqsRole",
            assumed_by=iam.ServicePrincipal("apigateway.amazonaws.com"),
            role_name="ApiGwV2ToSqsRole",
        )

        ApiGwToSqsRole.add_managed_policy(
            iam.ManagedPolicy.from_managed_policy_arn(
                self,
                "ApiGwPushCwPolicy",
                "arn:aws:iam::aws:policy/service-role/"
                "AmazonAPIGatewayPushToCloudWatchLogs",
            ),
        )

        # Lambda SQS Handler to Dynamo DB Role
        SqsHandlerLambdaExecutionRole = iam.Role(
            self,
            "SqsHandlerLambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            role_name="SqsHandlerLambdaExecutionRole",
        )

        SqsHandlerLambdaExecutionRole.attach_inline_policy(
            iam.Policy(
                self,
                "SqsHandlerInlinePolicy",
                statements=[
                    iam.PolicyStatement(
                        actions=[
                            "dynamodb:List*",
                            "dynamodb:DescribeReservedCapacity*",
                            "dynamodb:DescribeLimits",
                            "dynamodb:DescribeTimeToLive",
                            "dynamodb:Get*",
                            "dynamodb:PutItem",
                        ],
                        resources=[table.table_arn],
                    ),
                    # code for generating policy statement for bedrock inference
                    iam.PolicyStatement(
                        actions=[
                            "bedrock:InvokeModel",
                            "bedrock:InvokeModelWithResponseStream",
                        ],
                        resources=[
                            f"arn:aws:bedrock:{self._region}:{self._accountId}:inference-profile/{self.MODEL}",
                            f"arn:aws:bedrock:*::foundation-model/{self.MODEL.replace("us.", "")}",
                        ],
                    ),
                    iam.PolicyStatement(
                        actions=[
                            "logs:CreateLogGroup",
                            "logs:CreateLogStream",
                            "logs:PutLogEvents",
                        ],
                        resources=["*"],
                    ),
                ],
            )
        )

        # Authorization Lambda Role
        AuthorizationLambdaRole = iam.Role(
            self,
            "AuthorizationLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            role_name="AuthorizationLambdaRole",
        )

        AuthorizationLambdaRole.attach_inline_policy(
            iam.Policy(
                self,
                "AuthorizationInlinePolicy",
                statements=[
                    iam.PolicyStatement(
                        actions=[
                            "logs:CreateLogGroup",
                            "logs:CreateLogStream",
                            "logs:PutLogEvents",
                        ],
                        resources=["*"],
                    ),
                    iam.PolicyStatement(
                        actions=[
                            "secretsmanager:GetSecretValue",
                        ],
                        resources=[
                            secretUsername.secret_arn,
                            secretPassword.secret_arn,
                        ],
                    ),
                ],
            )
        )

        CGMLambdaExecutionRole = iam.Role(
            self,
            "CGMLambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            role_name="CGMLambdaExecutionRole",
        )

        CGMLambdaExecutionRole.attach_inline_policy(
            iam.Policy(
                self,
                "CGMHandlerInlinePolicy",
                statements=[
                    iam.PolicyStatement(
                        actions=[
                            "dynamodb:List*",
                            "dynamodb:DescribeReservedCapacity*",
                            "dynamodb:DescribeLimits",
                            "dynamodb:DescribeTimeToLive",
                            "dynamodb:Get*",
                            "dynamodb:PutItem",
                        ],
                        resources=[table2.table_arn],
                    ),
                    iam.PolicyStatement(
                        actions=[
                            "secretsmanager:GetSecretValue",
                        ],
                        resources=["*"],
                    ),
                    iam.PolicyStatement(
                        actions=[
                            "logs:CreateLogGroup",
                            "logs:CreateLogStream",
                            "logs:PutLogEvents",
                        ],
                        resources=["*"],
                    ),
                ],
            )
        )

        ##########################################################################
        #   Lambda Functions                                                     #
        ##########################################################################
        pandas_layer = lambda_.LayerVersion.from_layer_version_arn(
            self,
            "AWSSDKPandas-Python312-Arm64",
            "arn:aws:lambda:"
            + self._region
            + ":336392948345:layer:AWSSDKPandas-Python312-Arm64:14",
        )

        fnSqsHandler = lambda_.Function(
            self,
            "SqsHandlerFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("lambda/handler/"),
            role=SqsHandlerLambdaExecutionRole,
            architecture=lambda_.Architecture.ARM_64,
            timeout=Duration.seconds(33),
            layers=[pandas_layer],
            environment={
                "DYNAMO_DB_NAME": table.table_name,
                "BEDROCK_MODEL_ID": self.MODEL,
            },
            function_name="SqsHandlerFunction",
        )
        # Lambda - AuthorizerFunction
        fnAuthorizer = lambda_.Function(
            self,
            "AuthorizerFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.lambda_handler",
            architecture=lambda_.Architecture.ARM_64,
            code=lambda_.Code.from_asset("lambda/authorizer/"),
            timeout=Duration.seconds(33),
            role=AuthorizationLambdaRole,
            environment={
                "usernameSecretArn": secretUsername.secret_arn,
                "passwordSecretArn": secretPassword.secret_arn,
            },
            function_name="AuthorizerFunction",
        )
        # Lambda - AuthorizerFunction
        cgmLambda = lambda_.Function(
            self,
            "CGMLambdaFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="cgm.lambda_handler",
            architecture=lambda_.Architecture.ARM_64,
            code=lambda_.Code.from_asset("lambda/cgm/"),
            timeout=Duration.seconds(16),
            role=CGMLambdaExecutionRole,
            function_name="CGMLambdaFunction",
            layers=[self.create_dependencies_layer_cgm("cgm", "CGMLambdaFunction")],
        )

        aws_events.Rule(
            self,
            f"cgmCron",
            schedule=aws_events.Schedule.cron(
                minute="*/5", hour="*", month="*", week_day="*", year="*"
            ),
            targets=[aws_events_targets.LambdaFunction(handler=cgmLambda)],
            rule_name=f"cgmCronRule",
        )

        principal = iam.ServicePrincipal("apigateway.amazonaws.com")
        ApiGwToSqsRole.attach_inline_policy(
            iam.Policy(
                self,
                "ApiGwV2ToSqsInlinePolicy",
                statements=[
                    iam.PolicyStatement(
                        actions=[
                            "lambda:InvokeFunction",
                        ],
                        resources=[
                            fnSqsHandler.function_arn,
                            f"{fnSqsHandler.function_arn}:alias*",
                        ],
                    ),
                ],
            ),
        )

        ##########################################################################
        #   API GW                                                               #
        ##########################################################################

        # API GW v2 HTTP API
        Apigwv2 = apigwv2.CfnApi(
            self,
            "HttpToSqs",
            cors_configuration=apigwv2.CfnApi.CorsProperty(
                allow_credentials=False,
                allow_headers=["*"],
                allow_methods=["POST"],
                allow_origins=["*"],
                max_age=43200,
            ),
            name="HttpToSqs",
            protocol_type="HTTP",
        )

        # API GW v2 Stage
        stage = apigwv2.CfnStage(
            self,
            "HttpToSqsStage",
            api_id=Apigwv2.ref,
            stage_name="$default",
            auto_deploy=True,
            access_log_settings=apigwv2.CfnStage.AccessLogSettingsProperty(
                destination_arn=logGroup.attr_arn,
                format='{ "requestId":"$context.requestId", "ip": "$context.identity.sourceIp", "requestTime":"$context.requestTime", "httpMethod":"$context.httpMethod","routeKey":"$context.routeKey", "status":"$context.status","protocol":"$context.protocol", "responseLength":"$context.responseLength" }',
            ),
            default_route_settings={
                "throttlingBurstLimit": 10,  # Max burst requests
                "throttlingRateLimit": 5,  # Requests per second
            },
        )

        # API GW v2 Integration
        httpApiIntegSqsSendMessage = apigwv2.CfnIntegration(
            self,
            "httpApiIntegSqsSendMessage",
            api_id=Apigwv2.ref,
            integration_type="AWS_PROXY",
            integration_method="POST",
            payload_format_version="2.0",
            credentials_arn=ApiGwToSqsRole.role_arn,
            integration_uri=fnSqsHandler.function_arn,
        )

        # API GW v2 Authorizer
        authorizer = apigwv2.CfnAuthorizer(
            self,
            "ApiGwAuthorizer",
            api_id=Apigwv2.ref,
            authorizer_type="REQUEST",
            name="AuthorizerFunction",
            authorizer_uri="arn:aws:apigateway:"
            + self._region
            + ":lambda:path/2015-03-31/functions/"
            + fnAuthorizer.function_arn
            + "/invocations",
            authorizer_payload_format_version="2.0",
            authorizer_result_ttl_in_seconds=0,
            enable_simple_responses=True,
        )

        # API GW v2 Route
        HttpApiRoute = apigwv2.CfnRoute(
            self,
            "HttpApiRouteSqsSendMsg",
            api_id=Apigwv2.ref,
            route_key="POST /submit",
            target="/".join(["integrations", httpApiIntegSqsSendMessage.ref]),
            authorization_type="CUSTOM",
            authorizer_id=authorizer.attr_authorizer_id,
        )

        # Create Resource policy for SQS Handler Lambda, which only allows access from the created API GW
        fnAuthorizer.add_permission(
            "apigateway.amazonaws.com",
            principal=principal,
            source_arn="arn:aws:execute-api:"
            + self._region
            + ":"
            + self._accountId
            + ":"
            + Apigwv2.attr_api_id
            + "/authorizers/*",
        )

        ##########################################################################
        #   Output                                                               #
        ##########################################################################

        CfnOutput(
            self,
            "HttpApiEndpoint",
            description="API Endpoint",
            value=Apigwv2.attr_api_endpoint,
        )
