import React from 'react'
import {Navbar, Nav } from 'react-bootstrap'
import { LinkContainer } from 'react-router-bootstrap'

function Navigation() {
    return (
        <Navbar bg="light">
            <LinkContainer to="/" style={{ marginLeft: "30px" }}>
                <Navbar.Brand>APalysis</Navbar.Brand>
            </LinkContainer>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
                <Nav className="mr-auto">
                    <LinkContainer to="/featurehunt">
                        <Nav.Link>Feature Hunt</Nav.Link>
                    </LinkContainer>
                    <LinkContainer to="/about">
                        <Nav.Link>About</Nav.Link>
                    </LinkContainer>
                </Nav>
            </Navbar.Collapse>
        </Navbar>
    )
}

export default Navigation